# Mini LLM: Shakespeare Text Generation

**Student:** Chenrui Liu

**Student ID:** a1984046

## 1. Data and model structure

This project implements a character-level GPT based on the supplied *llm-from-scratch* tutorial [1]. Its objective is next-character prediction, trained from scratch using cross-entropy loss. The Shakespeare dataset contains 65 distinct characters. The first 90% of the text forms the training set and the remaining 10% forms the validation set. Sampled windows remain within their respective splits. The vocabulary follows the tutorial and uses the complete text's character inventory.

The baseline has six Transformer blocks, six attention heads, an embedding dimension of 384, and a context length of 256 characters. Token embeddings are added to learned absolute positional embeddings. Each block applies pre-layer normalization, causal multi-head attention, a residual connection, and a second normalized residual sublayer containing a feed-forward network. The feed-forward network expands the embedding dimension by a factor of four and uses GELU. A final layer normalization precedes the output projection, which shares weights with the token embedding table.

Character tokenization keeps the vocabulary small but requires the model to learn spelling. The causal mask prevents access to future characters; the context length limits the available history. Residual connections and normalization support optimization, while the number and width of blocks determine model capacity and computational cost.

With width $d=384$, each head has width $d/6=64$. Each block uses $4d^2+4d$ parameters for the query, key, value, and output projections, $8d^2+5d$ for the two feed-forward layers, and $4d$ for its two layer normalizations. Including token and position embeddings and the final normalization, the total is $(65+256)d+6(12d^2+13d)+2d=10{,}770{,}816$. Weight tying avoids a second $65d$ parameter table for the output.

## 2. Experimental design and modifications

The historical baseline completed 5,000 optimization steps. The initial size comparison used a maximum of 5,000 steps. The follow-up comparison uses a maximum of 10,000 steps and stops after five consecutive evaluations without a strictly lower validation loss. Evaluations occur every 100 completed updates, so five failures span 500 updates. A tie counts as a failure. All runs use batch size 64 and AdamW with weight decay 0.01. The learning rate increases to 0.001 over 100 warmup steps, then follows cosine decay toward 0.0001. The gradient norm is clipped at 1.0. The training seed is 42. The follow-up runs restart from random initialization because the cosine learning-rate horizon changes with the step ceiling. The 5,000-step runs remain recorded as a separate experiment. All reported losses are character-level cross-entropy (CE).

| Experiment | Layers / heads / width | Dropout | Hypothesis |
|---|---|---:|---|
| Baseline | 6 / 6 / 384 | 0 | Reference configuration |
| Tiny | 2 / 2 / 128 | 0.1 | Test lower capacity |
| Small | 4 / 4 / 256 | 0.1 | Test intermediate capacity |
| Medium | 6 / 6 / 384 | 0.1 | Test default size with regularization |

Dropout is added to embeddings, attention weights, and residual-branch outputs. The three size configurations change depth, head count, and width together. Their results describe overall configuration differences. The data, tokenizer, context length, batch size, learning-rate schedule, training seed, and evaluation windows are shared. Early stopping can produce different actual step and token budgets. All samples use each run's lowest-loss checkpoint. Medium also provides a comparison with the historical baseline at the same model size, with dropout and the stopping policy changed. The best-checkpoint selection rule is shared by both.

The baseline retains the tutorial's model and optimizer settings. The training implementation uses separate random-number generators for training and evaluation. At step 0 and every 100 completed steps through step 5,000, it estimates each split's loss over the same 20 batches in evaluation mode and records the aligned losses in JSON and CSV. It saves the checkpoint with the lowest observed validation loss for generation. It also saves the latest model, optimizer, and random-number generator states for resumption, along with the configuration and data fingerprint for compatibility checks. Validation is used for model selection, so an independent test set would be needed to estimate final generalization separately.

Generation uses Hamlet's first line, “To be, or not to be, that is the question:”, followed by a newline. Each continuation contains 800 new characters. Temperatures 0.6, 0.8, and 1.0 are compared with top-k fixed at 40 and seeds 42, 123, and 2026. Lower temperature concentrates the sampling distribution; higher temperature increases diversity. These settings affect generation rather than the checkpoint's validation loss.

## 3. Results and discussion

| Initial experiment | Parameters | Best CE | Best step | Actual steps | Character targets | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| Historical baseline | 10,770,816 | 1.5701 | 1,500 | 5,000 | 81,920,000 | 294.2 s |
| Tiny | 437,888 | 1.8335 | 4,800 | 5,000 | 81,920,000 | 32.8 s |
| Small | 3,241,728 | 1.4818 | 4,900 | 5,000 | 81,920,000 | 132.3 s |
| Medium | 10,770,816 | 1.4939 | 2,400 | 2,900 | 47,513,600 | 219.8 s |

![Baseline training and validation cross-entropy](runs/baseline_5080_v1/loss_curve.png)

The baseline ran on an NVIDIA RTX 5080 with PyTorch 2.11.0+cu128 from repository revision `cc08b02de2ab7a12c0d50af737ea24490f72862b`. It completed 5,000 optimizer steps and processed 81,920,000 character targets in sampled windows. The logged elapsed time includes evaluation through the last step. Validation cross-entropy fell from 4.3627 at initialization to 1.5701 nats per character at step 1,500. By step 5,000, training loss had fallen from 1.1861 to 0.1538, while validation loss rose to 4.0702. This divergence indicates overfitting on this split. The 51 evaluation points and both checkpoints are in `runs/baseline_5080_v1/`. All baseline samples use the step-1,500 checkpoint.

At temperature 0.8 with seed 42, the continuation begins “This quarrel for his death: but stood and body,” then includes “His deservate well his compassion,” and later introduces a speaker as “JOHN OF GARNT:”. The sample captures dialogue layout, recognizable words, and Shakespearean cues such as “thee.” The invented forms “deservate” and “nenly,” incomplete clauses, and the abrupt change of speaker show weaker spelling, syntax, and continuity. At the same seed, temperature 0.6 repeats “the” heavily in some lines, while temperature 1.0 produces forms such as “magred” and “nunss.” These observations describe individual samples. The full nine-sample grid and parameters are in `runs/baseline_5080_v1/samples_topk_40/`. Character-level training helps explain how recognizable local spelling patterns can coexist with invented words. During autoregressive generation the model conditions on its own sampled characters, and after 256 characters the earliest context is cropped, which limits access to the opening prompt.

The size comparison uses one training seed per configuration. The three generation seeds vary sampling only. Validation supports checkpoint and configuration selection. Further training seeds and independent evaluation would strengthen conclusions. GPU sharing with a Carla simulation was observed before this comparison, so recorded runtimes reflect the available resources during each run.

![Validation curves for the size comparison](runs/comparison_scale_d01_p5_5080_v1/validation_comparison.png)

Tiny and Small reached the 5,000-step ceiling, with best checkpoints at steps 4,800 and 4,900. Medium reached its best CE of 1.4939 at step 2,400. Its validation losses at steps 2,500 through 2,900 all exceeded that value, triggering early stopping at step 2,900. The learning-rate schedule retained its original 5,000-step horizon. The stopping rule saved 2,100 planned updates in this run. A later improvement beyond the stopping point remains unobserved.

Small achieved the lowest observed validation CE, 1.4818, a reduction of 0.0884 from the historical baseline's best value. Its difference from Medium was only 0.0121, so a single training seed gives limited evidence for ranking these two configurations. Tiny's higher CE indicates weaker next-character predictions under this budget. Medium improved on the historical baseline at the same size and used dropout in training. Early stopping controls the amount of computation, while best-checkpoint selection was already available in the baseline. Comparison of the best observed checkpoints supports the combined experimental procedure, and additional runs would help isolate stochastic variation.

The following text observations use temperature 0.8, top-k 40, and seed 42 for every model, chosen as a fixed comparison setting. Tiny produces “This the shad is not dise the no,” with repeated function words, invented spellings, and fragmented line breaks. Small begins “Yet look upon my natural courtesy:” but later produces “And now no father eyes are deliver'd,” and invented forms including “Frunkage” and “winfer.” Medium produces recognizable phrases such as “God grant you” and “On my son-in-law!” alongside grammatical failures including “Having been do no dukedom?” Both Small and Medium show more recognizable local phrases than Tiny in these samples, while abrupt speaker changes and unstable meaning weaken continuity as a soliloquy. The original baseline also contains invented words and incomplete clauses. These examples support local qualitative observations, with all 27 new samples retained for inspection. The lower CE alone cannot establish an across-the-board improvement in literary quality.

The 10,000-step follow-up keeps dropout at 0.1 and patience at five evaluations. Each configuration was trained again from initialization with a cosine schedule planned over 10,000 steps. Thus the comparison with the first round changes both the permitted number of updates and the learning-rate trajectory.

| Follow-up configuration | Best validation CE | Best step | Stop step | Stop reason | Character targets | Runtime |
|---|---:|---:|---:|---|---:|---:|
| Tiny | 1.7070 | 9,600 | 10,000 | max_steps | 163,840,000 | 65.3 s |
| Small | 1.4793 | 4,800 | 5,300 | early_stopping | 86,835,200 | 139.9 s |
| Medium | 1.4986 | 2,200 | 2,700 | early_stopping | 44,236,800 | 204.4 s |

![Follow-up validation loss curves](runs/comparison_scale_d01_p5_10k_5080_v1/validation_comparison.png)

Tiny improved from 1.8335 to 1.7070, but reached the cap with four evaluations since its best point. Further improvement remains possible. Small stopped at step 5,300 after its best point at 4,800, achieving 1.4793. Medium stopped at step 2,700 after its best point at 2,200, achieving 1.4986. Their differences from the first round were small, with Medium slightly worse. Small is the provisional choice for further tuning under the current procedure. Its validation CE is 0.0908 lower than the original baseline's best CE and it uses 3,241,728 parameters. The actual training budgets differ under early stopping, and results use a single training seed.

At temperature 0.8 and seed 42, follow-up Small produces invented words including “restrainess” and “regroned,” as well as “What, has you would be,” which is grammatically unstable. Medium produces “To committedly we multitude them” and “we have standed.” These continuations still switch speakers and struggle to sustain the opening soliloquy. Selecting Small by validation CE establishes a quantitative preference. A final language-quality comparison should use the same prompt, length, temperatures, top-k values, and sampling seeds for the original baseline and selected model, retaining every original sample.

## 4. Attention and positional encoding

For unmasked attention, define

$$
A(X)=\operatorname{softmax}(S)V,\qquad
S=\frac{QK^\top}{\sqrt{d_k}},\qquad
Q=XW_Q,\ K=XW_K,\ V=XW_V.
$$

Let $P$ be a permutation matrix. For input $PX$, the projections become $PQ,PK,PV$, and the score matrix becomes $PSP^\top$. Row-wise softmax commutes with simultaneous row and column permutation because reordering leaves each row's normalization denominator unchanged. Hence

$$
\operatorname{softmax}(PSP^\top)
=P\operatorname{softmax}(S)P^\top,
$$

and therefore

$$
A(PX)=P\operatorname{softmax}(S)P^\top PV=PA(X).
$$

Thus unmasked self-attention is permutation equivariant.

With fixed positional encoding $E$, define $F(X)=A(X+E)$. A counterexample shows that this property no longer holds generally. Take two positions, scalar embeddings, $W_Q=W_K=W_V=1$, $X=(0,0)^\top$, and $E=(0,1)^\top$. Direct calculation gives

$$
F(X)=\begin{pmatrix}1/2\\ e/(1+e)\end{pmatrix}.
$$

If $P$ swaps the positions, then $PX=X$, so $F(PX)=F(X)$. However, $PF(X)$ swaps two unequal entries. Consequently $F(PX)\ne PF(X)$. Fixed absolute positional encodings therefore remove the general equivariance guarantee.

The proof assumes unmasked attention. If attention instead adds a mask $M$ to the logits, permuting the input and the mask together preserves equivariance, because the new logits are $P(S+M)P^\top$. Keeping the mask fixed gives $PSP^\top+M$, so the same proof requires $PMP^\top=M$. A fixed causal mask allows position 2 to attend to position 1 while excluding the reverse edge. Swapping the two positions changes the permitted edges. For example, with two scalar tokens $(0,1)^\top$, $W_Q=W_K=0$ and $W_V=1$, causal attention averages each available prefix and returns $(0,1/2)^\top$. Feeding the swapped input returns $(1,1/2)^\top$, which differs from the swapped original output $(1/2,0)^\top$.

## Reference

[1] angelos-p. *Train Your Own LLM From Scratch*, Parts 1–5, revision `63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9`. [Source repository](https://github.com/angelos-p/llm-from-scratch/tree/63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9)
