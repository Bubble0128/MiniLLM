# MiniLLM

A character-level GPT trained from scratch on Shakespeare, for the MATH_X405 project.
The code follows [angelos-p/llm-from-scratch](https://github.com/angelos-p/llm-from-scratch/tree/63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9), Parts 1 to 5, and extends it with the experiments described in the report.

## Contents

| | |
|---|---|
| [Report_English.md](Report_English.md) | Report: model structure, improvements, generated text, proof |
| [Experiments_Detailed.md](Experiments_Detailed.md) | Detailed log of every experiment |
| [MiniLLM_Submission.ipynb](MiniLLM_Submission.ipynb) | Notebook with outputs: configuration, loss curve, generated text, comparison table |
| [generated_text/](generated_text/) | Generated continuations of the Hamlet soliloquy, in folders named after training data, model and sampling setting |
| [appendix_more_data/](appendix_more_data/More_Data_Experiment.md) | Appendix: the tuned models retrained with 28 more Shakespeare plays |

## Results

Validation cross-entropy in nats per character at the best checkpoint of each run. Lower is better.

| Model | Layers / heads / width | Parameters | Best validation CE |
|---|---|---:|---:|
| Tutorial baseline | 6 / 6 / 384 | 10,770,816 | 1.5701 |
| Small, dropout 0.1 | 4 / 4 / 256 | 3,241,728 | 1.4793 |
| Medium tuned, seeds 42 / 73 / 2026 | 6 / 6 / 384 | 10,795,776 | 1.4753 / 1.4791 / 1.4848 |
| **Final: Small tuned, seeds 42 / 73 / 2026** | 4 / 4 / 256 | 3,258,368 | 1.4648 / 1.4762 / 1.4696 |

The final model is Small with an untied output projection and weight decay 0.1.

## Repository layout

**Core modules**, used by the notebook:

| File | Purpose |
|---|---|
| `model.py` | `GPTConfig`, causal self-attention, MLP, Transformer blocks, GPT |
| `data.py` | Pinned dataset, character encoding, 90/10 split, optional extra training text |
| `word_data.py` | Word-level tokenizer and data |
| `train.py` | Training loop, learning-rate schedule, evaluation, early stopping, checkpoints, named configurations |
| `generate.py` | Temperature and top-k sampling |
| `analysis.py` | Loss curves, comparison tables, numerical check for the proof |

**Everything else:**

| Path | Purpose |
|---|---|
| `experiments/` | One script per experiment, grouped by training data: `original_data/`, `extended_data/` ([index](experiments/README.md)) |
| `results/all_runs.csv` | Every training run: data, model, settings, seed and validation CE |
| `figures/` | Plots used in the report |
| `tests/` | Unit tests |

## Running

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python train.py --experiment final
```

In the notebook, `EXPERIMENT` selects `baseline`, `tiny`, `small`, `medium`, `medium_tuned` or `final`, and
`RUN_TRAINING = True` trains that configuration; otherwise saved results are read from `runs/`.

All reported runs were trained on an NVIDIA RTX 5080. Validation data were used for model selection, so the
reported CE is not an independent test score.
