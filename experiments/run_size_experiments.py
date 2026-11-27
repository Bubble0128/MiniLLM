"""Train model-size or learning-rate comparisons with saved metrics and samples."""
import argparse
from dataclasses import asdict, replace
import fcntl
import json
from itertools import product
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import torch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # core modules are in the repository root
from analysis import compare_runs, plot_history, plot_validation_comparison
from data import download_shakespeare
from generate import save_samples
from train import experiment_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--run-tag', default='scale_d01_p5_10k_5080_v1')
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--experiments', nargs='+', choices=('tiny', 'small', 'medium'),
                        default=['tiny', 'small', 'medium'])
    parser.add_argument('--max-lrs', nargs='+', type=float,
                        help='Peak rates to compare, each with min_lr = 0.1 * max_lr')
    parser.add_argument('--reference-runs', nargs='*', default=[],
                        help='Completed runs to include in the comparison without training')
    args = parser.parse_args()
    if not args.run_tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_tag):
        parser.error('run-tag must contain only letters, digits, underscores and hyphens')
    root = Path(__file__).resolve().parent.parent
    runs = root / 'runs'
    comparison = runs / f'comparison_{args.run_tag}'
    comparison.mkdir(parents=True, exist_ok=True)
    lock = (comparison / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = download_shakespeare(root / 'data/shakespeare.txt')
    torch.set_num_threads(4)
    directories = [Path(path).resolve() for path in args.reference_runs]
    for reference in directories:
        if not json.loads((reference / 'summary.json').read_text())['complete']:
            raise ValueError(f'Reference run is incomplete: {reference}')
    failures = []
    for name, max_lr in product(args.experiments, args.max_lrs or [None]):
        suffix = '' if max_lr is None else f'_lr{max_lr:g}'.replace('.', 'p')
        run = runs / f'{name}{suffix}_{args.run_tag}'
        model_config, train_config = experiment_config(name)
        train_config = replace(train_config, max_steps=args.max_steps)
        if max_lr is not None:
            train_config = replace(train_config, max_lr=max_lr, min_lr=0.1 * max_lr)
        summary_path = run / 'summary.json'
        complete = False
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary['model_config'] != asdict(model_config) or summary['train_config'] != asdict(train_config):
                raise ValueError(f'Existing configuration differs: {run}')
            complete = summary['complete']
        if not complete:
            command = [sys.executable, '-u', 'train.py', '--experiment', name,
                       '--device', args.device, '--data', str(data), '--output', str(run),
                       '--max-steps', str(args.max_steps)]
            if max_lr is not None:
                command.extend(['--max-lr', str(train_config.max_lr),
                                '--min-lr', str(train_config.min_lr)])
            if (run / 'last.pt').exists():
                command.append('--resume')
            print(f'Training {name}: {run}', flush=True)
            with run.with_suffix('.log').open('a') as log:
                process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                run.with_suffix('.pid').write_text(str(process.pid) + '\n')
                code = process.wait()
            if code:
                failures.append(dict(run=str(run), exit_code=code))
                print(f'Training failed: {run}, exit code {code}. Continuing other runs.', flush=True)
                continue
        summary = json.loads(summary_path.read_text())
        history = json.loads((run / 'history.json').read_text())
        assert summary['complete']
        best = min(history, key=lambda row: row['val_loss'])
        assert best['step'] == summary['best_step']
        for label, expected in (('best', best), ('last', history[-1])):
            state = torch.load(run / f'{label}.pt', map_location='cpu', weights_only=True)
            assert state['step'] == expected['step']
            assert state['val_loss'] == expected['val_loss']
            assert state['history'] == history[:history.index(expected) + 1]
            assert state['train_config'] == summary['train_config']
            del state
        plt.close(plot_history(run))
        samples_dir = run / 'samples_topk_40'
        if not (samples_dir / 'samples.json').exists():
            records = save_samples(run / 'best.pt', samples_dir, device=args.device)
            assert len(records) == 9
            assert all(len(r['text']) == len(r['prompt']) + 800 for r in records)
            write_json(samples_dir / 'environment.json', dict(
                device=args.device, torch=str(torch.__version__),
                gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None))
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        print(f'Completed {name}: best={summary["best_val_loss"]:.6f} '
              f'at {summary["best_step"]}, stopped={summary["step"]}', flush=True)
        directories.append(run)
    if directories:
        compare_runs(directories, comparison)
        plt.close(plot_validation_comparison(directories, comparison))
    write_json(comparison / 'status.json', dict(complete=not failures,
               runs=[str(p) for p in directories], failures=failures))
    print(f'Batch finished with {len(failures)} training failures: {comparison}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
