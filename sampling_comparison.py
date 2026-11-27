"""Multi-seed nucleus-sampling comparison: character champion vs word model.

Why this exists: the first comparison used top_k=40 for both models, which keeps
62% of a 65-character vocabulary but only 0.56% of a 7122-word vocabulary, and
it had one seed per setting. Here both models are truncated by probability mass
(top-p) instead of by count, and every setting is sampled with many seeds.

Read-only with respect to existing runs: it loads two checkpoints, samples on
CPU, and writes only inside --out. It does not import or modify project code
other than the frozen model.py / word_data.py snapshot of the word run.
"""
import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

import torch

PROMPT = 'To be, or not to be, that is the question:'
GRIDS = {
    # Truncate by probability mass, so both vocabularies are cut comparably.
    'topp': [dict(name='T0.8_p0.9', temperature=0.8, top_p=0.9, top_k=0),
             dict(name='T1.0_p0.9', temperature=1.0, top_p=0.9, top_k=0),
             dict(name='T1.0_p1.0', temperature=1.0, top_p=1.0, top_k=0)],
    # Small top-k x temperature grid (top_k=5/10, temperature 0.6-1.2).
    'topk': [dict(name=f'T{t:g}_k{k}', temperature=t, top_p=1.0, top_k=k)
             for k in (5, 10) for t in (0.6, 0.8, 1.0, 1.2)],
}
CHAR_CKPT = 'runs/overnight_20261127T013704_336480Z/D2_champion_finetune1000_batch16/best.pt'
WORD_RUN = 'runs/word_tokenizer_20261127T034028Z'
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
LABEL_RE = re.compile(r"(?m)^([A-Z][A-Za-z ]+):$")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    from model import GPT, GPTConfig
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = GPT(GPTConfig(**checkpoint['model_config']))
    model.load_state_dict(checkpoint['model'])
    return model.eval(), checkpoint


@torch.inference_mode()
def sample_batch(model, prompt_ids, pieces, seeds, temperature, top_p, top_k,
                 target_chars, max_new_tokens, banned=()):
    """One row per seed, each with its own CPU generator, forward pass batched.

    Rows keep whole tokens until they have emitted >= target_chars characters.
    """
    block = model.config.block_size
    device = next(model.parameters()).device
    idx = torch.tensor([prompt_ids] * len(seeds), device=device)
    rngs = [torch.Generator().manual_seed(seed) for seed in seeds]
    emitted = [[] for _ in seeds]
    lengths = [0] * len(seeds)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block:])
        logits = logits[:, -1, :].double() / temperature
        for token in banned:
            logits[:, token] = -torch.inf
        if top_k:                                       # same rule as generate.py
            cutoff = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, -torch.inf)
        probs = logits.softmax(-1)
        if top_p < 1:
            ordered, order = probs.sort(dim=-1, descending=True)
            mass_before = ordered.cumsum(-1) - ordered
            ordered = ordered.masked_fill(mass_before >= top_p, 0.0)
            probs = torch.zeros_like(probs).scatter(1, order, ordered)
            probs = probs / probs.sum(-1, keepdim=True)
        probs = probs.cpu()                             # per-seed CPU generators keep seeds reproducible
        tokens = torch.stack([torch.multinomial(probs[row], 1, generator=rngs[row])
                              for row in range(len(seeds))]).to(device)
        idx = torch.cat([idx, tokens], dim=1)
        for row, token in enumerate(tokens[:, 0].tolist()):
            if lengths[row] < target_chars:
                emitted[row].append(token)
                lengths[row] += len(pieces[token])
        if min(lengths) >= target_chars:
            return emitted
    raise RuntimeError('max_new_tokens reached before every row had enough characters')


class Scorer:
    """Corpus-based surface statistics; none of them measures meaning."""

    def __init__(self, corpus, split_index):
        self.train = corpus[:split_index]
        self.vocab = {w.lower() for w in WORD_RE.findall(corpus)}
        train_words = [w.lower() for w in WORD_RE.findall(self.train)]
        self.grams = {n: {tuple(train_words[i:i + n]) for i in range(len(train_words) - n + 1)}
                      for n in (2, 3)}
        self.speakers = set(LABEL_RE.findall(corpus))

    def score(self, new):
        unk = new.count('<unk>')
        segments = [WORD_RE.findall(part) for part in new.replace('<unk>', ' \x00 ').split('\x00')]
        if new and new[-1].isalpha() and segments[-1]:
            segments[-1] = segments[-1][:-1]            # final word was cut by the length limit
        words = [w for segment in segments for w in segment]
        lower = [w.lower() for w in words]
        hits = {}
        for n in (2, 3):
            found = total = 0
            for segment in segments:                    # never bridge an <unk>
                segment = [w.lower() for w in segment]
                for i in range(len(segment) - n + 1):
                    total += 1
                    found += tuple(segment[i:i + n]) in self.grams[n]
            hits[n] = (found, total)
        repeats = sum(1 for segment in segments
                      for a, b in zip(segment, segment[1:]) if a.lower() == b.lower())
        longest = 0
        for start in range(len(new)):
            end = start + longest + 1
            while end <= len(new) and new[start:end] in self.train:
                longest = end - start
                end += 1
        labels = LABEL_RE.findall(new)
        return dict(words=len(words), unk=unk,
                    nonwords=[w for w in words if w.lower() not in self.vocab],
                    bigram=hits[2], trigram=hits[3],
                    distinct=len(set(lower)) / max(1, len(lower)),
                    adjacent_repeats=repeats, longest_copy=longest,
                    labels=len(labels),
                    invalid_labels=[label for label in labels if label not in self.speakers],
                    repeated_speaker=sum(1 for a, b in zip(labels, labels[1:]) if a == b))


def pool(rows):
    words = sum(r['words'] for r in rows)
    unk = sum(r['unk'] for r in rows)
    nonwords = [w for r in rows for w in r['nonwords']]
    ratio = lambda key: sum(r[key][0] for r in rows) / max(1, sum(r[key][1] for r in rows))
    per_sample = lambda key: [r[key][0] / max(1, r[key][1]) for r in rows]
    return dict(samples=len(rows), words=words, unk=unk,
                unk_rate=unk / max(1, words + unk),
                samples_with_unk=sum(1 for r in rows if r['unk']),
                nonwords=len(nonwords), nonword_rate=len(nonwords) / max(1, words),
                nonword_list=sorted(set(nonwords)),
                bigram_hit=ratio('bigram'), bigram_hit_sd=statistics.pstdev(per_sample('bigram')),
                trigram_hit=ratio('trigram'), trigram_hit_sd=statistics.pstdev(per_sample('trigram')),
                distinct_mean=statistics.mean(r['distinct'] for r in rows),
                distinct_sd=statistics.pstdev(r['distinct'] for r in rows),
                adjacent_repeats_per_100_words=100 * sum(r['adjacent_repeats'] for r in rows) / max(1, words),
                longest_copy_mean=statistics.mean(r['longest_copy'] for r in rows),
                longest_copy_max=max(r['longest_copy'] for r in rows),
                labels=sum(r['labels'] for r in rows),
                invalid_labels=sorted({l for r in rows for l in r['invalid_labels']}),
                repeated_speaker=sum(r['repeated_speaker'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--chars', type=int, default=400)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--grid', choices=sorted(GRIDS), default='topp')
    parser.add_argument('--char-ckpt', default=CHAR_CKPT, help='character checkpoint, relative to --root')
    parser.add_argument('--char-name', default='character', help='label for the character variant')
    parser.add_argument('--only', default='', help='comma-separated variant names to run')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--no-write', action='store_true')
    args = parser.parse_args()
    SETTINGS = GRIDS[args.grid]
    root = args.root.resolve()
    sys.path.insert(0, str(root / WORD_RUN / 'source'))
    from word_data import WordTokenizer
    torch.set_num_threads(args.threads)
    started = time.time()

    corpus = (root / 'data/shakespeare.txt').read_text(encoding='utf-8')
    split_index = int(0.9 * len(corpus))
    scorer = Scorer(corpus, split_index)
    seeds = list(range(1, args.seeds + 1))

    char_model, char_ckpt = load(root / args.char_ckpt)
    word_model, word_ckpt = load(root / WORD_RUN / 'word_model/best.pt')
    char_model, word_model = char_model.to(args.device), word_model.to(args.device)
    assert char_ckpt.get('tokenizer_type', 'character') == 'character'
    assert word_ckpt['tokenizer_type'] == 'word'
    assert char_ckpt['data_sha256'] == word_ckpt['data_sha256'] == hashlib.sha256(
        (root / 'data/shakespeare.txt').read_bytes()).hexdigest()
    tokenizer = WordTokenizer(word_ckpt['tokenizer']['vocabulary'],
                              word_ckpt['tokenizer']['min_frequency'])
    chars = char_ckpt['chars']
    char_prompt = [chars.index(c) for c in PROMPT]
    word_prompt = tokenizer.encode(PROMPT)
    assert 0 not in word_prompt and tokenizer.decode(word_prompt) == PROMPT

    variants = [
        dict(name=args.char_name, model=char_model, prompt=char_prompt, pieces=chars, banned=()),
        dict(name='word', model=word_model, prompt=word_prompt, pieces=tokenizer.vocabulary, banned=()),
        dict(name='word_no_unk', model=word_model, prompt=word_prompt, pieces=tokenizer.vocabulary, banned=(0,)),
    ]
    if args.only:
        variants = [v for v in variants if v['name'] in args.only.split(',')]
        assert variants, 'no variant matches --only'
    records, summary = [], []
    for setting in SETTINGS:
        for variant in variants:
            tick = time.time()
            emitted = sample_batch(variant['model'], variant['prompt'], variant['pieces'], seeds,
                                   setting['temperature'], setting['top_p'], setting['top_k'], args.chars,
                                   max_new_tokens=3 * args.chars, banned=variant['banned'])
            rows = []
            for seed, tokens in zip(seeds, emitted):
                raw = PROMPT + ''.join(variant['pieces'][t] for t in tokens)
                view = raw[:len(PROMPT) + args.chars]
                rows.append(scorer.score(view[len(PROMPT):]))
                records.append(dict(variant=variant['name'], setting=setting['name'],
                                    temperature=setting['temperature'], top_p=setting['top_p'],
                                    top_k=setting['top_k'],
                                    seed=seed, prompt=PROMPT, text=view, raw_text=raw,
                                    new_tokens=len(tokens), unknown_tokens=view.count('<unk>'),
                                    unk_banned=bool(variant['banned'])))
            summary.append(dict(variant=variant['name'], setting=setting['name'], **pool(rows)))
            print(f"{variant['name']:12s} {setting['name']}  {time.time() - tick:5.1f}s", flush=True)

    # Reference: real held-out text, same statistics, excerpts starting at a line start.
    val = corpus[split_index:]
    rows, oov, total = [], 0, 0
    for i in range(args.seeds):
        start = val.index('\n', int(i * (len(val) - 2 * args.chars) / args.seeds)) + 1
        excerpt = val[start:start + args.chars]
        rows.append(scorer.score(excerpt))
        found = WORD_RE.findall(excerpt)
        total += len(found)
        oov += sum(1 for w in found if w not in tokenizer.stoi)
    reference = dict(variant='real_validation_text', setting='-', **pool(rows))
    reference['words_outside_word_vocabulary_rate'] = oov / max(1, total)
    summary.append(reference)

    columns = ['variant', 'setting', 'samples', 'words', 'unk', 'unk_rate', 'samples_with_unk',
               'nonwords', 'nonword_rate', 'bigram_hit', 'bigram_hit_sd', 'trigram_hit',
               'trigram_hit_sd', 'distinct_mean', 'distinct_sd', 'adjacent_repeats_per_100_words',
               'longest_copy_mean', 'longest_copy_max', 'labels', 'repeated_speaker']
    print()
    for row in summary:
        print(f"{row['variant']:22s} {row['setting']:10s} words={row['words']:5d} "
              f"unk={row['unk']:3d} ({100 * row['unk_rate']:4.1f}%, in {row['samples_with_unk']:2d} samples) "
              f"nonword={row['nonwords']:3d} ({100 * row['nonword_rate']:4.1f}%) "
              f"2g={100 * row['bigram_hit']:4.1f}%±{100 * row['bigram_hit_sd']:.1f} "
              f"3g={100 * row['trigram_hit']:4.1f}%±{100 * row['trigram_hit_sd']:.1f} "
              f"distinct={row['distinct_mean']:.3f}±{row['distinct_sd']:.3f} "
              f"rep/100w={row['adjacent_repeats_per_100_words']:.2f} "
              f"copy mean={row['longest_copy_mean']:.1f} max={row['longest_copy_max']} "
              f"labels={row['labels']} invalid={row['invalid_labels']} same_speaker_twice={row['repeated_speaker']}")
    print(f"real text: words outside the word vocabulary = {100 * reference['words_outside_word_vocabulary_rate']:.1f}%")
    print(f"total {time.time() - started:.0f}s")

    if args.no_write:
        for record in records[:3]:
            print('-----', record['variant'], record['setting'], 'seed', record['seed'])
            print(record['text'])
        return
    out = args.out.resolve()
    assert out.is_dir() and not any(p.name != Path(__file__).name for p in out.iterdir()), \
        '--out must be a directory holding only this script'
    (out / 'samples').mkdir()
    for setting in SETTINGS:
        for variant in variants:
            body = [f"prompt={PROMPT!r} temperature={setting['temperature']} top_p={setting['top_p']} "
                    f"top_k={setting['top_k']} "
                    f"new_characters={args.chars} unk_banned={bool(variant['banned'])}\n"]
            for record in records:
                if record['variant'] == variant['name'] and record['setting'] == setting['name']:
                    body.append(f"\n===== seed {record['seed']} =====\n{record['text']}\n")
            (out / 'samples' / f"{variant['name']}__{setting['name']}.txt").write_text(
                ''.join(body), encoding='utf-8')
    (out / 'samples.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    with (out / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)
    provenance = dict(
        purpose='Multi-seed regeneration for a character-vs-word sample comparison.',
        grid=args.grid, prompt=PROMPT, settings=SETTINGS, seeds=seeds, new_characters=args.chars, device=args.device,
        threads=args.threads, torch=torch.__version__, python=sys.version.split()[0],
        checkpoints={
            args.char_name: dict(path=str(root / args.char_ckpt), sha256=sha256(root / args.char_ckpt),
                              step=char_ckpt['step'], val_loss=char_ckpt['val_loss']),
            'word': dict(path=str(root / WORD_RUN / 'word_model/best.pt'),
                         sha256=sha256(root / WORD_RUN / 'word_model/best.pt'),
                         step=word_ckpt['step'], val_loss=word_ckpt['val_loss'])},
        source_snapshot=str(root / WORD_RUN / 'source'),
        script_sha256=sha256(__file__),
        notes=[
            'top_p keeps the smallest set of tokens whose probability mass reaches p; p=1.0 disables it.',
            'top_k keeps the k most likely tokens (0 disables it): k=5/10 is 8%/15% of 65 characters but 0.07%/0.14% of 7122 word tokens.',
            'word_no_unk masks token 0 (<unk>) at sampling time; the model itself is unchanged.',
            'Word rows keep whole tokens until >=400 new characters, then text is cut to exactly 400 new characters.',
            'A final word cut by the length limit is excluded from the statistics.',
            'nonword = not in the full corpus vocabulary (case-insensitive); n-gram hits are against the training split.',
            'These are surface statistics; they do not measure grammar or meaning.',
            'One prompt only; it is not in the corpus. Results may differ for other prompts.'],
        elapsed_seconds=time.time() - started)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'written to {out}')


if __name__ == '__main__':
    main()
