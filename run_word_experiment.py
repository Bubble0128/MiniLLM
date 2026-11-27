"""One word-tokenizer experiment and matched-prompt qualitative samples.

Run from a frozen source snapshot under a systemd user service. No polling.
Word CE includes unknown words and layout tokens, and must not be ranked
against character CE. The larger vocabulary and longer text per context
are intrinsic confounds and recorded explicitly.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import traceback

import torch

from data import download_shakespeare
from generate import generate, load_model
from model import GPTConfig
from train import TrainConfig, train, write_json
from word_data import WordData, WordTokenizer


@torch.inference_mode()
def generate_words(model, tokenizer, prompt, temperature=0.8, top_k=40, seed=123,
                   target_characters=400, max_new_tokens=600):
    """Keep whole emitted tokens; save a separate 400-character comparison view."""
    ids = tokenizer.encode(prompt)
    if 0 in ids:
        raise ValueError('Prompt has unknown words; refusing a silently changed prompt')
    idx = torch.tensor([ids], device=next(model.parameters()).device)
    rng = torch.Generator().manual_seed(seed)
    continuation, emitted = '', []
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.config.block_size:])
        logits = logits[:, -1, :].float() / temperature
        if top_k:
            cutoff = torch.topk(logits, min(top_k, len(tokenizer.vocabulary))).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, -torch.inf)
        token = torch.multinomial(logits.softmax(-1).cpu(), 1, generator=rng).to(idx.device)
        idx = torch.cat([idx, token], dim=1)
        emitted.append(token.item())
        continuation += tokenizer.vocabulary[token.item()]
        if len(continuation) >= target_characters:
            break
    return prompt + continuation, emitted


def main(batch, root):
    torch.set_num_threads(4)
    plan = json.loads((batch / 'plan.json').read_text())
    state = dict(complete=False, stage='training', failures=[], plan=str(batch / 'plan.json'))
    write_json(batch / 'status.json', state)
    run = batch / 'word_model'
    try:
        result = train(download_shakespeare(root / 'data/shakespeare.txt'), run,
                       GPTConfig(**plan['model_config']), TrainConfig(**plan['train_config']),
                       device='cuda')
        if not result['complete']:
            raise RuntimeError('Training incomplete')
        for label in ('best', 'last'):
            checkpoint = torch.load(run / f'{label}.pt', map_location='cpu', weights_only=True)
            expected = result['best_step'] if label == 'best' else result['step']
            if checkpoint['step'] != expected or checkpoint['tokenizer_type'] != 'word':
                raise RuntimeError('Checkpoint verification failed')
            del checkpoint
        state.update(stage='generation', training_complete=True, best_word_CE=result['best_val_loss'])
        write_json(batch / 'status.json', state)
        samples = batch / 'samples'
        samples.mkdir()
        records = []
        prompt = 'To be, or not to be, that is the question:'
        for kind, path in [('word', run / 'best.pt'), ('character', Path(plan['character_champion']))]:
            try:
                model, checkpoint = load_model(path, 'cuda', expected_tokenizer=kind)
                tokenizer = WordTokenizer(checkpoint['tokenizer']['vocabulary'],
                                          checkpoint['tokenizer']['min_frequency']) if kind == 'word' else None
                for temperature in (0.6, 0.8, 1.0):
                    try:
                        if kind == 'word':
                            text, emitted = generate_words(model, tokenizer, prompt, temperature=temperature)
                        else:
                            text = generate(model, prompt, checkpoint['chars'], max_new_tokens=400,
                                            temperature=temperature, top_k=40, seed=123)
                            emitted = None
                        stem = f'{kind}_temperature_{temperature:g}_topk_40_seed_123'
                        (samples / f'{stem}.txt').write_text(text, encoding='utf-8')
                        # Exact raw prefix, never edited or cleaned up.
                        view = text[:len(prompt) + 400]
                        (samples / f'{stem}_400chars.txt').write_text(view, encoding='utf-8')
                        records.append(dict(tokenizer_type=kind, file=f'{stem}.txt',
                                            comparison_view=f'{stem}_400chars.txt', text=text,
                                            temperature=temperature, top_k=40, seed=123, prompt=prompt,
                                            new_characters=len(text) - len(prompt),
                                            new_tokens=len(emitted) if emitted is not None else 400,
                                            emitted_token_ids=emitted,
                                            unknown_tokens=emitted.count(0) if emitted is not None else 0,
                                            checkpoint=str(path), checkpoint_step=checkpoint['step']))
                    except Exception as error:
                        state['failures'].append(dict(stage='sample', kind=kind, temperature=temperature,
                                                      error=repr(error)))
                        traceback.print_exc()
                    write_json(samples / 'samples.json', records)
                del model, checkpoint
                torch.cuda.empty_cache()
            except Exception as error:
                state['failures'].append(dict(stage='generation', kind=kind, error=repr(error)))
                traceback.print_exc()
        char_summary = json.loads((Path(plan['character_champion']).parent / 'summary.json').read_text())
        rows = []
        for kind, summary in [('character', char_summary), ('word', result)]:
            config = summary['model_config']
            rows.append(dict(tokenizer=kind, best_CE=summary['best_val_loss'],
                             CE_unit='nats/character' if kind == 'character' else 'nats/word_or_layout_token',
                             comparable_CE=False, best_step=summary['best_step'],
                             trained_steps=summary['step'], parameters=summary['parameters'],
                             vocab_size=config['vocab_size'], block_size_tokens=config['block_size']))
        with (batch / 'tokenizer_comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        state.update(stage='finished', complete=not state['failures'], generated_samples=len(records))
    except Exception as error:
        state.update(stage='failed', complete=False, error=repr(error))
        traceback.print_exc()
        raise
    finally:
        write_json(batch / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    main(args.batch.resolve(), args.root.resolve())
