"""Sequential course ablations; isolated child processes, no polling, no plots.

Each child blocks its parent until exit (subprocess.run), without monitoring.
Historical metrics are preserved; common fixed evaluation windows enable B/T
comparisons. CSV configuration columns expose differing historical budgets.
"""
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from data import download_shakespeare
from model import GPTConfig
from train import TrainConfig, train, write_json


def completed_runs(root):
    """Read only genuine completed experiments with matching data and checkpoints."""
    records = []
    for path in sorted((root / 'runs').rglob('summary.json')):
        try:
            s = json.loads(path.read_text())
            if not s.get('complete') or not (path.parent / 'best.pt').exists():
                continue
            if s.get('tokenizer_type', 'character') != 'character':
                continue  # Per-token CE is not comparable across tokenizers.
            if s.get('data_sha256') != '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed':
                continue
            if not math.isfinite(s['best_val_loss']):
                continue
            records.append((path.parent, s))
        except (OSError, ValueError, KeyError):
            print(f'Cannot use summary: {path}', flush=True)
    return records


def select_lr(root):
    eligible = [(p, s) for p, s in completed_runs(root)
                if p.name.startswith('medium')
                and s['model_config']['n_layer'] == 6
                and s['model_config']['n_embd'] == 384
                and s['model_config']['dropout'] == 0.1
                and s['train_config']['max_lr'] in (3e-4, 1e-3, 3e-3)]
    if not eligible:
        raise RuntimeError('No completed Medium learning-rate experiments')
    return min(eligible, key=lambda item: item[1]['best_val_loss'])


def export_csv(root, batch, failures):
    """Include historical, completed, incomplete and failed experiments explicitly."""
    rows = []
    for path in sorted((root / 'runs').rglob('summary.json')):
        try:
            s = json.loads(path.read_text())
            history = json.loads((path.parent / 'history.json').read_text())
            best = min(history, key=lambda h: h['val_loss'])
            mc, tc = s['model_config'], s['train_config']
            ce = s['best_val_loss']
            row = dict(experiment=str(path.parent.relative_to(root)),
                       status='complete' if s.get('complete') else 'incomplete',
                       config=json.dumps(dict(model=mc, train=tc, parent=s.get('parent')), sort_keys=True),
                       best_step=s['best_step'], best_Train_CE=best['train_loss'],
                       best_Val_CE=ce, best_Val_PPL=math.exp(ce) if ce < 700 else 'inf',
                       final_Val_CE=s['last_val_loss'], final_step=s['step'],
                       parameters=s['parameters'], elapsed_seconds=s['elapsed_seconds'],
                       tokens_seen=s.get('tokens_seen'),
                       eval_T=tc.get('eval_block_size') or mc['block_size'],
                       eval_B=tc.get('eval_batch_size') or tc['batch_size'], error='')
            rows.append(row)
        except Exception as error:
            rows.append(dict(experiment=str(path.parent.relative_to(root)), status='unreadable', error=str(error)))
    by_name = {row['experiment']: row for row in rows}
    for failure in failures:
        name = failure['experiment']
        row = by_name.get(name)
        if row is None:
            row = dict(experiment=name)
            rows.append(row)
            spec_path = (root / name).with_suffix('.spec.json')
            if spec_path.exists():
                row['config'] = spec_path.read_text()
        row.update(status='failed', error=failure['error'])
    fields = ['experiment', 'status', 'config', 'best_step', 'best_Train_CE',
              'best_Val_CE', 'best_Val_PPL', 'final_Val_CE', 'final_step',
              'parameters', 'elapsed_seconds', 'tokens_seen', 'eval_T', 'eval_B', 'error']
    temporary = batch / 'overnight_summary.csv.tmp'
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(batch / 'overnight_summary.csv')
    # Convenient top-level copy; original historical files are never modified.
    shutil.copy2(batch / 'overnight_summary.csv', root / 'overnight_summary.csv')


def worker(spec_path):
    """One subprocess per training run, allowing CUDA recovery after failures."""
    import torch
    from data import CharacterData
    from generate import load_model
    from train import evaluate
    torch.set_num_threads(4)
    spec = json.loads(Path(spec_path).read_text())
    mc, tc = GPTConfig(**spec['model']), TrainConfig(**spec['training'])
    result = train(spec['data'], spec['output'], mc, tc, device='cuda',
                   init_checkpoint=spec.get('init_checkpoint'))
    if not result['complete']:
        raise RuntimeError('Training did not reach its intended stop')
    for name in ('best', 'last'):
        checkpoint = torch.load(Path(spec['output']) / f'{name}.pt', map_location='cpu', weights_only=True)
        expected = result['best_step'] if name == 'best' else result['step']
        if checkpoint['step'] != expected or checkpoint['train_config'] != asdict(tc):
            raise RuntimeError('Checkpoint verification failed')
        del checkpoint
    if mc.block_size != tc.eval_block_size:
        # Secondary metric measures the benefit of the full 512-character context.
        model, _ = load_model(Path(spec['output']) / 'best.pt', 'cuda')
        native = evaluate(model, CharacterData(spec['data']),
                          replace(tc, eval_block_size=mc.block_size,
                                  eval_batch_size=tc.batch_size), 'cuda')
        write_json(Path(spec['output']) / 'native_context_evaluation.json',
                   dict(scores=native, block_size=mc.block_size, batch_size=tc.batch_size,
                        note='Secondary native-context score; selection uses common T=256/B=64.'))


def scan_generation(checkpoint, output):
    """Ten unique settings (the two requested scans overlap at k=20)."""
    import torch
    from generate import generate, load_model
    torch.set_num_threads(4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, state = load_model(checkpoint, 'cuda')
    settings = sorted({(t, 20) for t in (0.6, 0.8, 1.0, 1.2)} |
                      {(t, k) for t in (0.6, 0.8) for k in (1, 10, 20, 40)})
    records, errors = [], []
    prompt = 'To be, or not to be, that is the question:'
    for temperature, k in settings:
        try:
            # Explicit argmax for the requested greedy control, including ties.
            effective_temperature = 0 if k == 1 else temperature
            text = generate(model, prompt, state['chars'], max_new_tokens=400,
                            temperature=effective_temperature, top_k=k, seed=123)
            filename = f'temperature_{temperature:g}_topk_{k}_seed_123.txt'
            (output / filename).write_text(text, encoding='utf-8')
            records.append(dict(file=filename, text=text, prompt=prompt, temperature=temperature,
                                effective_temperature=effective_temperature, top_k=k, seed=123,
                                decoding='greedy' if k == 1 else 'multinomial', max_new_tokens=400,
                                checkpoint=str(checkpoint), checkpoint_step=state['step'],
                                validation_loss=state['val_loss'], model_config=state['model_config'],
                                data_sha256=state['data_sha256']))
        except Exception as error:
            errors.append(dict(temperature=temperature, top_k=k, error=repr(error)))
            traceback.print_exc()
        write_json(output / 'samples.json', records)
        write_json(output / 'errors.json', errors)
    if errors:
        raise RuntimeError(f'{len(errors)} generation settings failed')


def main(root):
    root = root.resolve()
    lock = (root / 'overnight.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    chosen_path, chosen = select_lr(root)
    best_lr = chosen['train_config']['max_lr']
    tag = datetime.now(timezone.utc).strftime('overnight_%Y%m%dT%H%M%S_%fZ')
    batch = root / 'runs' / tag
    batch.mkdir()
    snapshot = batch / 'source'
    snapshot.mkdir()
    for name in ('run_overnight.py', 'run_overnight.sh', 'model.py', 'train.py', 'data.py', 'generate.py'):
        shutil.copy2(root / name, snapshot / name)
    data = download_shakespeare(root / 'data/shakespeare.txt')
    state = dict(batch=str(batch), pid=__import__('os').getpid(), BEST_LR=best_lr,
                 lr_reference=str(chosen_path), stage='starting', failures=[], runs=[], complete=False,
                 source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()},
                 notes=[
                     'All new from-scratch runs: 5000 steps, warmup 150, no early stopping.',
                     'Common selection CE: fixed T=256 B=64, 20 batches, eval seeds 12345/12346.',
                     'Context512 additionally records native-context CE; fixed training B*T=16384.',
                     'Historical warmup/budgets/LRs differ: this is not a fully matched factorial design.',
                     'A chooses among its three successful new runs; if all fail, use the historical LR reference.',
                     'SGD exception: max_lr=0.1/min_lr=0.01, momentum=0.9, same cosine/warmup/weight decay.',
                     'SGD is one tuned setting, not proof that AdamW is necessary.',
                     'D seed check uses winner architecture/LR with current 5000-step/150-warmup budget; historical budgets may differ.',
                     'Fine-tuning loads parent weights, resets AdamW state and step count, uses seed42/43 and warmup150.',
                     'Validation selects configurations; no independent test-set claim.'
                 ])
    def persist():
        write_json(batch / 'status.json', state)
        write_json(root / 'overnight_latest.json', state)
        export_csv(root, batch, state['failures'])
    base_model = GPTConfig(dropout=0.1)
    base_train = TrainConfig(max_steps=5000, max_lr=best_lr, min_lr=0.1 * best_lr,
                             warmup_steps=150, patience=0, eval_batch_size=64, eval_block_size=256)
    print(f'BEST_LR={best_lr:g}; source={chosen_path}; output={batch}', flush=True)
    persist()

    def launch(name, mc, tc, parent=None):
        output = batch / name
        spec = dict(model=asdict(mc), training=asdict(tc), data=str(data), output=str(output),
                    init_checkpoint=str(parent) if parent else None)
        spec_path = batch / f'{name}.spec.json'
        write_json(spec_path, spec)
        state['stage'] = name
        persist()
        print(f'START {name}: {json.dumps(spec)}', flush=True)
        try:
            with (batch / f'{name}.log').open('x') as log:
                result = subprocess.run([sys.executable, '-u', str(snapshot / 'run_overnight.py'),
                                         '--worker', str(spec_path)], cwd=root, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f'Child exited {result.returncode}; see {name}.log')
            summary = json.loads((output / 'summary.json').read_text())
            if not summary.get('complete'):
                raise RuntimeError('Incomplete run')
            state['runs'].append(str(output))
            print(f'DONE {name}: CE={summary["best_val_loss"]}', flush=True)
            return output, summary
        except Exception as error:
            state['failures'].append(dict(experiment=str(output.relative_to(root)), error=repr(error)))
            traceback.print_exc()
            return None
        finally:
            persist()

    group_a = []
    for name, dropout, wd in [('A1_d01_wd01', .1, .1), ('A2_d02_wd001', .2, .01), ('A3_d02_wd01', .2, .1)]:
        result = launch(name, replace(base_model, dropout=dropout), replace(base_train, weight_decay=wd))
        if result:
            group_a.append(result)
    selected_path, selected = min(group_a, key=lambda item: item[1]['best_val_loss']) if group_a else (chosen_path, chosen)
    best_do, best_wd = selected['model_config']['dropout'], selected['train_config']['weight_decay']
    state.update(BEST_DO=best_do, BEST_WD=best_wd, regularization_reference=str(selected_path))
    mc, tc = replace(base_model, dropout=best_do), replace(base_train, weight_decay=best_wd)
    launch('B_context512_batch32', replace(mc, block_size=512), replace(tc, batch_size=32))
    launch('C1_residual_scale05', replace(mc, residual_scale=0.5), tc)
    launch('C2_no_weight_tying', replace(mc, weight_tying=False), tc)
    launch('C3_depth4_heads6_width384', replace(mc, n_layer=4), tc)
    launch('C4_sgd_lr01_momentum09', mc, replace(tc, optimizer_name='sgd', max_lr=0.1, min_lr=0.01))

    def champion():
        # Exclude failed overnight children even if they wrote a complete training summary.
        failed = {item['experiment'] for item in state['failures']}
        pool = [(p, s) for p, s in completed_runs(root)
                if str(p.relative_to(root)) not in failed
                and (s['train_config'].get('eval_block_size') or s['model_config']['block_size']) == 256
                and (s['train_config'].get('eval_batch_size') or s['train_config']['batch_size']) == 64]
        return min(pool, key=lambda item: item[1]['best_val_loss'])

    winner_path, winner = champion()
    state['pre_D_champion'] = str(winner_path)
    winner_model = GPTConfig(**winner['model_config'])
    winner_train = TrainConfig(**winner['train_config'])
    launch('D1_champion_seed73', winner_model,
           replace(winner_train, seed=73, max_steps=5000, warmup_steps=150, patience=0,
                   grad_clip=1.0, eval_batch_size=64, eval_block_size=256))
    # Re-evaluate selection after the seed check; fine-tune the best available checkpoint.
    winner_path, winner = champion()
    state['fine_tune_parent'] = str(winner_path)
    launch('D2_champion_finetune1000_batch16', GPTConfig(**winner['model_config']),
           replace(TrainConfig(**winner['train_config']), optimizer_name='adamw', seed=42,
                   max_steps=1000, max_lr=2e-4, min_lr=2e-5, warmup_steps=150,
                   batch_size=16, patience=0, grad_clip=1.0, eval_batch_size=64, eval_block_size=256),
           winner_path / 'best.pt')
    winner_path, winner = champion()
    state.update(stage='E_generation', champion=str(winner_path), champion_CE=winner['best_val_loss'])
    persist()
    try:
        with (batch / 'generation.log').open('x') as log:
            subprocess.run([sys.executable, '-u', str(snapshot / 'run_overnight.py'),
                            '--generate', str(winner_path / 'best.pt'),
                            '--output', str(batch / 'generation')], cwd=root,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    except Exception as error:
        state['failures'].append(dict(experiment=str((batch / 'generation').relative_to(root)), error=repr(error)))
        traceback.print_exc()
    state.update(stage='finished', complete=not state['failures'], finished=True)
    persist()
    print(f'FINISHED: {len(state["failures"])} failures; summary={batch / "overnight_summary.csv"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--generate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.generate:
        scan_generation(args.generate, args.output)
    else:
        main(args.root)
