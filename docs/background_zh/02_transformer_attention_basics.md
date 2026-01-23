# 02｜Transformer/Attention 最小必要知识（面向推理小白）

> 目标：不学完整 Transformer 课程，只掌握**读懂推理系统**必须的概念：
>
> - 什么是 token / embedding / hidden states
> - attention 在做什么（Q/K/V）
> - 为什么会有 KV cache
> - 为什么“解码阶段”可以只算 1 个 token

---

## 1. token、embedding、hidden states 是什么？

### 1.1 token

Tokenizer 会把文本切成 token，并把每个 token 映射成一个整数 id：

- 文本：`"Hello world"`
- token ids：`[123, 456, ...]`

LLM 实际处理的是 token ids 序列。

### 1.2 embedding

token id 本身是离散的整数，模型需要把它变成向量（连续值）才能做矩阵运算：

- `embedding_table[vocab_size, hidden_dim]`
- `x_t = embedding_table[token_id_t]`

### 1.3 hidden state

Transformer 每层都会把输入向量变换成新的向量。每个 token 在每层都有一个向量表示：

- hidden state 形状常见为 `[seq_len, hidden_dim]`（或带 batch 维度）

---

## 2. Attention 在干什么？（只讲推理需要的）

对于某一层，给定当前的 hidden states \(X\)：

- \(Q = X W_Q\)
- \(K = X W_K\)
- \(V = X W_V\)

attention 输出大致是：

\[
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
\]

### 2.1 causal attention（自回归约束）

在生成任务中，第 \(t\) 个 token 只能看见 \(\le t\) 的位置，不能偷看未来。

这会导致一个关键性质：

> 当你在 decode 阶段生成第 \(t\) 个 token 时，只需要为“新 token”算一个新的 query（\(q_t\)），并让它和历史的 \(K,V\) 做 attention。

---

## 3. 为什么会有 KV cache？

想象你已经算完前 \(t-1\) 个 token。第 \(t\) 步生成时，如果你重新把 1..t 的序列整段再过一遍 Transformer，会极其浪费。

关键观察：

- 过去 token 的 \(K,V\) 在后续步骤不会变（模型参数固定、输入序列固定）
- 所以后续 decode 只需要“复用历史的 \(K,V\)”

因此推理系统会缓存每层的 \(K,V\)：

- **KV cache**：保存所有历史 token 的 K/V
- decode 每步只新增 1 个 token 的 K/V，并 append 到 cache

这会把每步计算从“对整段序列做 attention”变成“对一个 query 做 attention”，极大加速生成。

---

## 4. prefill vs decode：从 attention 的视角理解

### 4.1 prefill

prefill 要把 prompt 的所有 token 都过一遍：

- 产生整段的 K/V（写入 KV cache）
- 计算量大，尤其是 attention 需要处理长序列

### 4.2 decode

decode 每步只处理 1 个新 token：

- 只新增 1 个 token 的 K/V
- attention 的 query 长度为 1，key/value 长度为当前序列长度
- 更适合做高度优化（固定形态、频繁重复）

---

## 5. 在 Mini-SGLang 中你会在哪里看到这些概念？

- KV cache 的写入：
  - `python/minisgl/kvcache/mha_pool.py` 的 `store_kv(...)`
  - 内部调用 `python/minisgl/kernel/store.py` 的 `store_cache(...)`

- attention 的实现（选择不同 backend）：
  - `python/minisgl/attention/fa3.py`
  - `python/minisgl/attention/fi.py`

如果你下一步想理解“为什么 KV cache 会吃显存、怎么管理”，建议读：

- `04_kv_cache_basics.md`


