# Generated text

Continuations of the prompt `To be, or not to be, that is the question:` (Hamlet, Act 3, Scene 1).
Each file holds 20 samples (seeds 1-20, 400 new characters) for one model and one sampling setting.
File names give the temperature and top-k, for example `T0.8_k10`.

| Folder | Model |
|---|---|
| `final_character/` | Final character-level model (untied output projection, validation CE 1.4752) |
| `baseline_character/` | Tutorial baseline (validation CE 1.5701) |
| `word/` | Word-level model; `<unk>` marks an out-of-vocabulary word |
| `word_no_unk/` | Word-level model with `<unk>` masked during sampling |

`summary_final_and_word.csv` and `summary_baseline.csv` hold the statistics quoted in the report.
`sampling_comparison.py` in the repository root produced these files.
