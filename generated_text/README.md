# Generated text

Continuations of the prompt `To be, or not to be, that is the question:` (Hamlet, Act 3, Scene 1).
Each file holds 20 samples (seeds 1-20, 400 new characters) for one model and one sampling setting.
File names give the temperature and top-k, for example `T0.8_k10`. All samples were generated on the GPU.

| Folder | Model |
|---|---|
| `final_character/` | Final model: Small tuned, seed 42 (validation CE 1.4648) |
| `medium_tuned_character/` | Medium tuned, seed 42 (validation CE 1.4753) |
| `baseline_character/` | Tutorial baseline (validation CE 1.5701) |
| `word/` | Word-level model; `<unk>` marks an out-of-vocabulary word |
| `word_no_unk/` | Word-level model with `<unk>` masked during sampling |

The `summary_*.csv` files hold the statistics quoted in the report.
`experiments/sampling_comparison.py` produced these files.
