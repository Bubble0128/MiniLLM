# Experiments

Scripts are grouped by the training data they use. Run them from the repository root, for example
`python experiments/seed_check.py final 2026`. Every run is written to
`runs/<training data>/<tokenizer>_L<layers>-H<heads>-D<width>_ctx<context>_drop<dropout>_lr<peak rate>_wd<weight decay>_<tied|untied>_steps<ceiling>_seed<seed>/`,
so the folder name states the data, the model and the settings. `results/all_runs.csv` lists every run with its validation CE.

## `original_data/`: the original Shakespeare file (1.0M training characters)

| Script | Experiment | Detailed log |
|---|---|---|
| `size_and_learning_rate.py` | Tiny, Small and Medium models; learning-rate comparison | 3.1, 3.2 |
| `ablations_medium.py`, `ablations_medium.sh` | Ablations on Medium: dropout, weight decay, context length, residual scaling, untied output projection, depth, optimizer, second seed, continued training | 3.3, 3.4 |
| `ablations_medium_followup.py`, `continued_training_followup.py` | Further context-length and dropout runs; second continuation | 3.3, 3.4 |
| `small_tuned.py` | Small tuned (final model): untied output projection and weight decay 0.1, with single-change controls | 3.4 |
| `word_tokenizer.py` | Word-level tokenizer | 3.6 |

The three ablation drivers ran as timestamped batches; they are kept as the record of how those runs were launched.

## `extended_data/`: original file plus 28 more plays (4.4M training characters)

| Script | Experiment | Detailed log |
|---|---|---|
| `prepare_extended_data.py` | Downloads the Project Gutenberg Complete Works and builds the extra training text | Appendix |
| `train_with_extended_data.py` | Trains the tuned models and the word-level models with the extra text | Appendix |

## Either data set

| Script | Purpose | Detailed log |
|---|---|---|
| `seed_check.py` | Repeats a named configuration with another training seed | 3.4 |
| `sampling_comparison.py` | Temperature and top-k sampling with 20 seeds per setting; produces `generated_text/` | 3.5, 3.6, Appendix |
