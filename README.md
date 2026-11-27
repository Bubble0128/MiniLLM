# MiniLLM

Character-level GPT trained from scratch on Shakespeare for the MATH_X405 project.
The code follows [angelos-p/llm-from-scratch](https://github.com/angelos-p/llm-from-scratch/tree/63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9), Parts 1 to 5.

| Deliverable | Location |
|---|---|
| Report | [Report_English.md](Report_English.md) |
| Notebook with outputs | [MiniLLM_Submission.ipynb](MiniLLM_Submission.ipynb) ([open in Colab](https://colab.research.google.com/github/Bubble0128/MiniLLM/blob/main/MiniLLM_Submission.ipynb)) |
| Generated text | [generated_text/](generated_text/) |
| Appendix: follow-up experiment with more training text | [appendix_more_data/](appendix_more_data/More_Data_Experiment.md) |

## Results

Validation cross-entropy in nats per character, best checkpoint of each run. Lower is better.

| Model | Layers / heads / width | Parameters | Best validation CE |
|---|---|---:|---:|
| Tutorial baseline | 6 / 6 / 384 | 10,770,816 | 1.5701 |
| Small, dropout 0.1 | 4 / 4 / 256 | 3,241,728 | 1.4793 |
| Medium tuned, seeds 42 / 73 / 2026 | 6 / 6 / 384 | 10,795,776 | 1.4753 / 1.4791 / 1.4848 |
| **Final: Small tuned, seeds 42 / 73 / 2026** | 4 / 4 / 256 | 3,258,368 | 1.4648 / 1.4762 / 1.4696 |

The final model is Small with an untied output projection and weight decay 0.1. The report describes every experiment.

## Files

| Path | Purpose |
|---|---|
| `model.py` | `GPTConfig`, causal attention, MLP, Transformer blocks, GPT |
| `data.py` | Pinned dataset, character encoding, 90/10 split, optional extra training text |
| `word_data.py` | Word-level tokenizer and data |
| `train.py` | Training loop, named configurations (`experiment_config`), checkpoints |
| `generate.py` | Temperature and top-k sampling |
| `analysis.py` | Loss curves, comparison tables, numerical check for the proof |
| `experiments/` | One script per experiment: size and learning-rate comparison, ablations, word tokenizer, tuned Small model, seed checks, sampling comparison, extra training text |
| `results/` | Result records of the ablation batch |
| `appendix_more_data/` | Separate follow-up: the tuned models retrained with 28 more Shakespeare plays |
| `figures/`, `generated_text/` | Plots used in the report; generated samples and their statistics |
| `build_submission_notebook.py` | Regenerates the notebook source |
| `tests/` | Unit tests |

## Running

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python train.py --experiment final
python experiments/run_extended_data.py final   # downloads and converts the extra text on first use
```

In the notebook, `EXPERIMENT` selects `baseline`, `tiny`, `small`, `medium`, `medium_tuned` or `final`.
Set `RUN_TRAINING = True` to train; otherwise it reads existing results from `runs/`.
All reported runs were trained on an NVIDIA RTX 5080. Validation data were used for model selection, so the reported CE is not an independent test score.

The code in this repository was written with AI assistance.
