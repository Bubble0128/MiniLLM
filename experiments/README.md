# Experiments

One script per experiment in the report. Run them from the repository root, for example
`python experiments/run_seed_check.py final 2026`. Results are written to `runs/`.

| Script | Experiment | Report |
|---|---|---|
| `run_size_experiments.py` | Tiny, Small and Medium models; learning-rate comparison | 3.1, 3.2 |
| `run_overnight.py`, `run_overnight.sh` | Ablations on Medium: dropout, weight decay, context length, residual scaling, untied output projection, depth, optimizer, second seed, continued training | 3.3, 3.4 |
| `run_extra.py`, `run_extra_e1.py` | Further context-length and dropout runs; second continuation | 3.3, 3.4 |
| `run_small_untied.py` | Small tuned: untied output projection and weight decay 0.1, with single-change controls | 3.4 |
| `run_seed_check.py` | Any named configuration with another training seed | 3.4 |
| `sampling_comparison.py` | Temperature and top-k sampling with 20 seeds per setting; produces `generated_text/` | 3.5, 3.6 |
| `run_word_experiment.py` | Word-level tokenizer | 3.6 |
| `prepare_extended_data.py` | Builds extra training text from the Project Gutenberg Complete Works | Appendix |
| `run_extended_data.py` | Trains the tuned models and the word-level models with the extra text | Appendix |
