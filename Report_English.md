Draft — complete the results and discussion after running the experiments.

# Mini LLM: Shakespeare Text Generation

**Student:** [TO COMPLETE]

**Student ID:** [TO COMPLETE]

## 1. Data and model structure

This project implements a character-level GPT based on the supplied *llm-from-scratch* tutorial [1]. Its objective is next-character prediction, trained from scratch using cross-entropy loss. The Shakespeare dataset contains 65 distinct characters. The first 90% of the text forms the training set and the remaining 10% forms the validation set. Sampled windows remain within their respective splits. The vocabulary follows the tutorial and uses the complete text's character inventory.

The baseline has six Transformer blocks, six attention heads, an embedding dimension of 384, and a context length of 256 characters. Token embeddings are added to learned absolute positional embeddings. Each block applies pre-layer normalization, causal multi-head attention, a residual connection, and a second normalized residual sublayer containing a feed-forward network. The feed-forward network expands the embedding dimension by a factor of four and uses GELU. A final layer normalization precedes the output projection, which shares weights with the token embedding table.

Character tokenization keeps the vocabulary small but requires the model to learn spelling. The causal mask prevents access to future characters; the context length limits the available history. Residual connections and normalization support optimization, while the number and width of blocks determine model capacity and computational cost.

## 2. Experimental design and modifications

All full experiments use 5,000 optimization steps, batch size 64, and AdamW with weight decay 0.01. The learning rate increases to 0.001 over 100 warmup steps, then follows cosine decay toward 0.0001. The gradient norm is clipped at 1.0. The training seed is 42.

| Experiment | Layers / heads / width | Dropout | Hypothesis |
|---|---|---:|---|
| Baseline | 6 / 6 / 384 | 0 | Reference configuration |
| Dropout | 6 / 6 / 384 | 0.1 | Regularization may reduce overfitting |
| Small | 4 / 4 / 256 | 0 | Lower capacity may improve generalization and reduce runtime |

Dropout is added to embeddings, attention weights, and residual-branch outputs. The small-model experiment changes overall capacity, so its outcome cannot be attributed independently to depth, head count, or width. Other training settings remain fixed.

Engineering changes include fixed evaluation windows, separate training and evaluation random-number generators, aligned loss records, best-checkpoint selection, and resumable checkpoints. Every 100 steps, training and validation losses are estimated over 20 batches in evaluation mode. The checkpoint with the lowest observed validation loss is selected. Validation therefore supports model selection rather than constituting an independent test set.

Generation uses Hamlet's first line, “To be, or not to be, that is the question:”, followed by a newline. Each continuation contains 800 new characters. Temperatures 0.6, 0.8, and 1.0 are compared with top-k fixed at 40 and seeds 42, 123, and 2026. Lower temperature concentrates the sampling distribution; higher temperature increases diversity. These settings affect generation rather than the checkpoint's validation loss.

## 3. Results and discussion

| Experiment | Best validation loss | Best step | Runtime |
|---|---:|---:|---:|
| Baseline | [TO COMPLETE] | [TO COMPLETE] | [TO COMPLETE] |
| Dropout | [TO COMPLETE] | [TO COMPLETE] | [TO COMPLETE] |
| Small | [TO COMPLETE] | [TO COMPLETE] | [TO COMPLETE] |

**Learning curves:** [TO COMPLETE: insert the loss figure and discuss convergence and the training–validation gap.]

**Generated language:** [TO COMPLETE: quote comparable continuations and discuss spelling, grammar, coherence, repetition, and resemblance to Shakespeare.]

**Design choices:** [TO COMPLETE: connect observed improvements or regressions to regularization, model capacity, and temperature; justify the final configuration using the recorded evidence.]

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

This theorem concerns unmasked attention. The implemented GPT uses a fixed causal mask, which also prevents equivariance under arbitrary position permutations.

## Reference

[1] angelos-p. *Train Your Own LLM From Scratch*, Parts 1–5, revision `63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9`. [Source repository](https://github.com/angelos-p/llm-from-scratch/tree/63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9)
