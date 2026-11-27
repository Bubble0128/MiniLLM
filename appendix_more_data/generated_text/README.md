# Generated text, models trained with the extra text

Same prompt, seeds and file naming as `generated_text/` in the repository root.

| Folder | Model |
|---|---|
| `small_character/` | Small tuned + extra text, seed 42 |
| `medium_character/` | Medium tuned + extra text, seed 42 |
| `word_medium/` | Word-level tokenizer, Medium tuned architecture, seed 42 |
| `word_small/` | Word-level tokenizer, Small tuned architecture, seed 42 |

The statistics in `summary_*.csv` score every sample against the original and the extra text together.
`summary_final_model_original_text.csv` scores the final model of the report in the same way.
