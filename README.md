# MiniLLM

Character-level GPT trained from scratch on Shakespeare for the MATH_X405 project.
The code follows [angelos-p/llm-from-scratch](https://github.com/angelos-p/llm-from-scratch/tree/63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9), Parts 1 to 5.

| Deliverable | Location |
|---|---|
| Report | [Report_English.md](Report_English.md) |
| Notebook with outputs | [MiniLLM_Submission.ipynb](MiniLLM_Submission.ipynb) ([open in Colab](https://colab.research.google.com/github/Bubble0128/MiniLLM/blob/main/MiniLLM_Submission.ipynb)) |
| Generated text | [generated_text/](generated_text/) |

## Results

Validation cross-entropy in nats per character, best checkpoint of each run.

| Model | Layers / heads / width | Parameters | Best validation CE |
|---|---|---:|---:|
| Tutorial baseline | 6 / 6 / 384 | 10,770,816 | 1.5701 |
| Small, dropout 0.1 | 4 / 4 / 256 | 3,241,728 | 1.4793 |
| Medium tuned, seeds 42 / 73 / 2026 | 6 / 6 / 384 | 10,795,776 | 1.4753 / 1.4791 / 1.4848 |
| **Final: Small tuned, seeds 42 / 73 / 2026** | 4 / 4 / 256 | 3,258,368 | 1.4648 / 1.4762 / 1.4696 |

The final model is Small with an untied output projection and weight decay 0.1. The report describes every experiment.

## Files

| File | Purpose |
|---|---|
| `model.py` | `GPTConfig`, causal attention, MLP, Transformer blocks, GPT |
| `data.py`, `word_data.py` | Pinned dataset, character and word tokenizers, 90/10 split |
| `train.py` | Training loop, named configurations (`experiment_config`), checkpoints |
| `generate.py`, `sampling_comparison.py` | Temperature and top-k sampling, multi-seed comparison |
| `analysis.py` | Loss curves, comparison tables, numerical check for the proof |
| `run_size_experiments.py`, `run_overnight.py`, `run_extra.py`, `run_extra_e1.py`, `run_word_experiment.py`, `run_small_untied.py`, `run_seed_check.py` | Experiment drivers |
| `prepare_extended_data.py`, `run_extended_data.py` | Extra training text from the Complete Works (Project Gutenberg) and the runs that use it |
| `overnight_summary.csv` | Results of the ablation batch |
| `build_submission_notebook.py`, `build_notebook.py` | Regenerate the notebooks from source |
| `MiniLLM_Colab.ipynb` | Chinese walkthrough notebook |
| `tests/` | Unit tests |

## Running

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python train.py --experiment final
```

In the notebook, `EXPERIMENT` selects `baseline`, `tiny`, `small`, `medium`, `medium_tuned` or `final`.
Set `RUN_TRAINING = True` to train; otherwise it reads existing results from `runs/`.
All reported runs were trained on an NVIDIA RTX 5080. Validation data were used for model selection, so the reported CE is not an independent test score.

The code in this repository was written with AI assistance.
