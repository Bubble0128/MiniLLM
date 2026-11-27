# Generated text, models trained with the extra text

Same prompt, sampling seeds and naming as `generated_text/` in the repository root:

```
generated_text/extended_data/<model and training settings>/temperature<T>_topk<k>.txt
```

| Folder under `extended_data/` | Model |
|---|---|
| `char_L4-H4-D256_ctx256_drop0.1_lr0.001_wd0.1_untied_steps10000-earlystop5_seed42` | Small tuned + extra text, seed 42 |
| `char_L6-H6-D384_ctx256_drop0.1_lr0.0003_wd0.1_untied_steps10000_seed42` | Medium tuned + extra text, seed 42 |
| `word_L6-H6-D384_ctx256_drop0.1_lr0.0003_wd0.1_untied_steps10000-earlystop5_seed42` | Word-level tokenizer, Medium tuned architecture + extra text, seed 42 |
| `word_L4-H4-D256_ctx256_drop0.1_lr0.001_wd0.1_untied_steps10000-earlystop5_seed42` | Word-level tokenizer, Small tuned architecture + extra text, seed 42 |

`statistics.csv` in each folder scores the samples against the original and the extra text together.
`original_data/final_model_statistics_scored_with_extra_text.csv` scores the final model of the report in the same way.
