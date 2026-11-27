# Generated text

Continuations of the prompt `To be, or not to be, that is the question:` (Hamlet, Act 3, Scene 1).

The path of every file states how it was produced:

```
generated_text/<training data>/<model and training settings>/temperature<T>_topk<k>.txt
```

`<model and training settings>` is the name of the training run: tokenizer, layers-heads-width, context length, dropout,
peak learning rate, weight decay, tied or untied output projection, step ceiling (with early-stopping patience) and seed.
Each file holds 20 samples (sampling seeds 1 to 20, 400 new characters). `statistics.csv` in each folder holds the
figures quoted in the report. All models here were trained on the original text only.

| Folder under `original_data/` | Model |
|---|---|
| `char_L4-H4-D256_ctx256_drop0.1_lr0.001_wd0.1_untied_steps10000-earlystop5_seed42` | Final model: Small tuned, seed 42 |
| `char_L6-H6-D384_ctx256_drop0.1_lr0.0003_wd0.1_untied_steps5000_seed42` | Medium tuned, seed 42 |
| `char_L6-H6-D384_ctx256_drop0_lr0.001_wd0.01_tied_steps5000_seed42` | Tutorial baseline |
| `word_L6-H6-D384_ctx256_drop0.1_lr0.0003_wd0.1_untied_steps5000-earlystop5_seed42` | Word-level tokenizer; `<unk>` marks an out-of-vocabulary word |
| `word_L6-H6-D384_ctx256_drop0.1_lr0.0003_wd0.1_untied_steps5000-earlystop5_seed42__unk-masked` | The same word-level model with `<unk>` masked during sampling |

Produced by `experiments/sampling_comparison.py`.
