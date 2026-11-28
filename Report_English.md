# Mini LLM: Shakespeare Text Generation

**Student:** Chenrui Liu

**Student ID:** a1984046

## 1. Model structure

The model is the character-level GPT of the *llm-from-scratch* tutorial [1], trained from scratch to predict the next character. The Shakespeare file has 65 distinct characters; the first 90% (1.0 million characters) is the training split and the last 10% the validation split. All losses are cross-entropy (CE) in nats per character on the validation split.

Each character is mapped to a token embedding, to which a learned positional embedding is added. The result passes through a stack of Transformer blocks. A block applies layer normalization, causal multi-head self-attention and a residual connection, then layer normalization, a feed-forward network of four times the width with GELU, and a second residual connection. A final layer normalization and a linear layer give a score for each of the 65 characters.

| Choice | Baseline (tutorial) | Final model | What it controls |
|---|---|---|---|
| Tokens | characters | characters | Small vocabulary, but spelling has to be learned |
| Context length | 256 | 256 | How much earlier text can influence a prediction, fewer than 50 words |
| Layers / heads / width | 6 / 6 / 384 | 4 / 4 / 256 | Capacity |
| Parameters | 10,770,816 | 3,258,368 | |
| Output projection | tied to the embedding | untied | Whether one matrix serves as input embedding and output classifier |
| Dropout | 0 | 0.1 | Regularization |
| Weight decay | 0.01 | 0.1 | Regularization |
| Checkpoint used | best validation CE | best validation CE, early stopping | When training ends |

The causal mask lets a position attend only to earlier positions. Training can therefore score all 256 positions of a window at once with the true text as input, while generation must feed the model its own sampled characters. Both models use AdamW with a peak learning rate of 0.001, 100 warmup steps, cosine decay, gradient clipping at 1.0 and batch size 64. Every setting is a named field of `GPTConfig` or `TrainConfig`, and each modification is marked by a comment in the code.

## 2. Model changes and tuning

**What was lacking.** The baseline memorizes the training text. Its validation CE is lowest at step 1,500 (1.5701); by step 5,000 the training CE has fallen to 0.15 while the validation CE has risen to 4.07.

![Training and validation cross-entropy of the baseline](figures/baseline_loss_curve.png)

**Path to the final model.**

| Step | Model | Validation CE |
|---|---|---:|
| Baseline | 6 / 6 / 384, tutorial settings | 1.5701 |
| 1 | + dropout 0.1 and early stopping | 1.4986 |
| 2 | Smaller model, 4 / 4 / 256 | 1.4793 |
| 3 | + untied output projection and weight decay 0.1 (final model) | 1.4648 |

**What was tried.** Each row changes one setting and keeps the rest fixed; the value that was kept is in bold. Medium is the 6 / 6 / 384 model and Small the 4 / 4 / 256 model.

| Setting | Model | Values tried | Validation CE |
|---|---|---|---|
| Layers / heads / width | | 2 / 2 / 128, **4 / 4 / 256**, 6 / 6 / 384 | 1.7070, **1.4793**, 1.4986 |
| Peak learning rate | Small | 0.0003, **0.001**, 0.003 | 1.5387, **1.4793**, 1.4807 |
| Peak learning rate | Medium | **0.0003**, 0.001, 0.003 | **1.4796**, 1.4986, 2.1294 |
| Dropout | Medium | **0.1**, 0.2 | **1.4996**, 1.5930 |
| Weight decay | Small | 0.01, **0.1** | 1.4793, **1.4728** |
| Output projection | Small | tied, **untied** | 1.4793, **1.4745** |
| Output projection | Medium | tied, **untied** | 1.4996, **1.4753** |
| Context length | Medium | **256**, 512 | **1.4996**, 1.5418 |
| Residual branch scale | Medium | **1.0**, 0.5 | **1.4996**, 1.5587 |
| Layers | Medium | **6**, 4 | **1.4996**, 1.5501 |
| Optimizer | Medium | **AdamW**, SGD with momentum | **1.4996**, 3.0460 |

*Regularization.* Dropout on the embeddings, attention weights and residual branches, together with stopping once the validation CE no longer improves, removes the divergence of the baseline. This follows the lectures on generalization: overfitting is the dominant source of error here. Dropout 0.2 was too strong for the step budget, and a weight decay of 0.1 gave a small further gain.

*Model size and learning rate.* With one million training characters, the model with 3.2 million parameters does at least as well as the one with 10.8 million, and the smallest model under-fits. The two larger models need different step sizes: Small is best at 0.001, while Medium is best at 0.0003 and fails to train at 0.003. The suitable step size depends on the model, as the lectures on optimization suggest.

*Architecture.* The tutorial shares one 65 × width matrix between the token embedding and the output layer. Giving the output layer its own matrix costs 16,640 parameters on Small and lets the two roles differ: one describes a character as context, the other scores it as a prediction. It helped on both models. A longer context, damped residual branches and fewer layers were all worse.

*Optimizer.* SGD with momentum, at one tuned setting, stayed far behind AdamW within the same number of steps, which shows how much the adaptive step sizes help here.

The final model combines the kept values on Small: dropout 0.1, untied output projection and weight decay 0.1.

![Training and validation cross-entropy of the final model](figures/final_model_loss_curve.png)

**Small against Medium.** The same two changes were applied to Medium, with its own best peak learning rate of 0.0003, and both models were trained with three seeds.

| Model | Layers / heads / width | Parameters | Seed 42 | Seed 73 | Seed 2026 | Mean |
|---|---|---:|---:|---:|---:|---:|
| Small tuned (final model) | 4 / 4 / 256 | 3,258,368 | 1.4648 | 1.4762 | 1.4696 | 1.4702 |
| Medium tuned | 6 / 6 / 384 | 10,795,776 | 1.4753 | 1.4791 | 1.4848 | 1.4797 |

Small is lower by 0.0095 on average, but runs that differ only in the seed vary by up to 0.011, so the two models are equal within the noise. A model three times larger brings no benefit on this amount of data, and Small tuned was chosen as the final model because it reaches the same CE with 30% of the parameters. The gain over the baseline, 0.10 nats per character (per-character perplexity 4.81 to 4.35), is about nine times the seed variation. The remaining gap between training CE (1.13) and validation CE (1.47) points to the amount of data as the limit.

## 3. Generated text

The prompt is the first line of Hamlet's soliloquy, “To be, or not to be, that is the question:”. Each setting was sampled with 20 seeds and 400 new characters. The table uses top-k 10 and reports the share of generated words that do not occur anywhere in the corpus, and the ratio of distinct words to all words in a sample, which is 0.745 for real held-out text.

| Temperature | Invented words, baseline | Invented words, final | Distinct-word ratio, final |
|---:|---:|---:|---:|
| 0.6 | 3.4% | 1.3% | 0.661 |
| 0.8 | 6.0% | 3.0% | 0.742 |
| 1.0 | 9.5% | 4.4% | 0.766 |
| 1.2 | 14.7% | 7.9% | 0.822 |

The final model invents roughly half as many words as the baseline, or fewer, at every setting, so the lower CE shows up mainly as better spelling. One complete continuation by the final model follows (temperature 0.8, top-k 10, seed 1; the first line is the prompt):

```text
To be, or not to be, that is the question:
Suppose of the sun that I will be my soul,
That invirced in the story and some comfort,
For when you with the world-hanging in your shiles,
To greet me and put up his majesty.

DUKE VINCENTIO:
It is no more at hid, if it be as it,
Too much weakness mine, that he that shall do,
In the heavens, the noble poor impatience
His son: which shall you are you have been:
Then holy we shall say you short wi
```

Two of its 81 new words are invented (“invirced”, “shiles”). A speaker label appears in the correct form and the lines have the length of verse, but no clause carries a clear meaning. A baseline sample at the same setting contains more invented words, for example “serivity” and “stopherd”.

**Temperature and top-k.** Temperature divides the scores before the softmax. At 0.6 the model rarely misspells but repeats common phrases, and its vocabulary is less varied than real text. At 1.2 it is more varied than real text and 7.9% of its words are invented. Top-k keeps only the k most likely characters; because a misspelling usually begins with an unlikely character, lowering k from 10 to 5 cuts invented words from 4.4% to 2.3% at temperature 1.0. Temperature 0.8 with top-k 10 matches the variety of real text most closely.

**How the text resembles English.** Most words are real, function words appear in plausible order, speaker labels and line breaks follow the layout of a play, and many lines have the length of a verse line. Each of these depends only on the previous few characters or words, which is what next-character prediction over a short window rewards.

**How it does not.** Clauses lose their grammar after a few words, speakers change without reason and no topic is sustained, so the continuation never reads as a soliloquy. The choices in Section 1 explain this:

- Character tokens make spelling something to be learned, hence the invented words, and more of them at higher temperature.
- A context of 256 characters is fewer than 50 words, so nothing earlier can shape the text.
- One million characters is little data. *Hamlet* is not in the corpus, so the model can only continue in the general style of the plays it saw.
- The model is trained on true text but generates from its own samples, so an early slip is carried forward.

A word-level tokenizer was also tried. It cannot misspell, but 8% to 16% of its word positions were the unknown-word token and its text was far more repetitive (distinct-word ratio 0.40 to 0.57), so the character model was kept.

## 4. Attention and positional encoding

The lecture on attention uses the scaled dot-product form of Vaswani et al. [2], $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$. In self-attention the queries, keys and values are linear maps of the same input $X$, which has one row per position. For unmasked self-attention, define

$$
A(X)=\mathrm{softmax}(S)V,\qquad
S=\frac{QK^\top}{\sqrt{d_k}},\qquad
Q=XW_Q,\ K=XW_K,\ V=XW_V.
$$

**Claim.** $A(PX)=PA(X)$ for every permutation matrix $P$.

**Proof.** For input $PX$ the projections become $PQ$, $PK$, $PV$, so the score matrix becomes $PQ(PK)^\top/\sqrt{d_k}=PSP^\top$. The softmax acts on each row separately, and permuting the entries of a row does not change its normalizing sum, so

$$
\mathrm{softmax}(PSP^\top)=P\,\mathrm{softmax}(S)\,P^\top .
$$

Since $P^\top P=I$,

$$
A(PX)=P\,\mathrm{softmax}(S)\,P^\top PV=P\,\mathrm{softmax}(S)\,V=PA(X).
$$

**With a positional encoding the property fails.** Let $E$ be a fixed positional encoding and $F(X)=A(X+E)$. Take two positions, scalar embeddings, $W_Q=W_K=W_V=1$, $X=(0,0)^\top$ and $E=(0,1)^\top$. Then $X+E=(0,1)^\top$, the score matrix has rows $(0,0)$ and $(0,1)$, and

$$
F(X)=\begin{pmatrix}1/2\\ e/(1+e)\end{pmatrix}.
$$

Let $P$ swap the two positions. Then $PX=X$, so $F(PX)=F(X)$, while $PF(X)$ has its two unequal entries exchanged. Hence $F(PX)\ne PF(X)$: once a fixed positional encoding is added, the input is no longer treated equivariantly, which is what allows the model to use word order.

The proof concerns unmasked attention. The causal mask of the GPT is also fixed to positions and breaks the property for the same reason.

## 5. Additional experiment: more training text

Section 2 ends with the amount of data as the limit. As an additional experiment, 28 further Shakespeare plays from Project Gutenberg were added to the training split, which gives 4.4 times more training text with the validation split unchanged. Small tuned and Medium tuned then reach a validation CE of 1.29 to 1.33, a larger improvement than all the changes of Section 2 together, and they invent fewer words. The model and the results of this report use the original file only. The additional experiment is described in the appendix, `appendix_more_data/More_Data_Experiment.md`.

## Further material

`results/all_runs.csv` lists all 41 training runs. `Experiments_Detailed.md` describes every experiment in full. `generated_text/` holds every generated sample.

## References

[1] angelos-p. *Train Your Own LLM From Scratch*, Parts 1–5, revision `63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9`. [Source repository](https://github.com/angelos-p/llm-from-scratch/tree/63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9)

[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser and I. Polosukhin. *Attention Is All You Need*. NeurIPS 2017.
