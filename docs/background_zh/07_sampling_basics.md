# 07｜采样（Sampling）基础：temperature/top-k/top-p（面向推理小白）

> 目标：你能理解“模型输出的 logits 是什么”，以及常见采样参数如何改变生成结果，并知道一些在线服务中常见的坑。

---

## 1. logits、概率分布与 next token

模型 forward 的输出（对每个位置）通常是一个向量：

- `logits[vocab_size]`

它不是概率，但可以通过 softmax 变成概率：

\[
p_i = \frac{e^{\text{logits}_i}}{\sum_j e^{\text{logits}_j}}
\]

生成 next token 就是从这个分布里“选一个 id”。

---

## 2. 最简单：greedy（贪心）

取概率最大的 token：

- `argmax(logits)`

优点：稳定、可复现  
缺点：容易重复/刻板/陷入局部最优

---

## 3. temperature：控制随机性

temperature \(T\) 的直觉是“把 logits 拉伸/压缩”：

- \(\text{logits}' = \text{logits} / T\)

效果：

- \(T \to 0\)：分布更尖锐，趋近 greedy
- \(T = 1\)：原分布
- \(T > 1\)：更随机

常见坑：

- 很多系统把 `temperature=0` 作为 greedy 的约定（不做随机采样）

---

## 4. top-k：只在前 k 个 token 中采样

做法：

1. 取 logits 最大的 k 个 token
2. 只在这 k 个上做 softmax + 采样

效果：

- 限制随机性来源，避免采到很离谱的 token

---

## 5. top-p（nucleus sampling）：只在累计概率达到 p 的集合中采样

做法：

1. 按概率从大到小排序
2. 取最小集合，使累计概率 ≥ p
3. 在集合内归一化后采样

top-p 会让“候选集合大小”自适应：

- 分布很尖锐时，集合很小
- 分布很平坦时，集合更大

---

## 6. repetition / frequency penalty（扩展概念）

很多生产系统会加入“避免重复”的惩罚项，例如：

- 对已出现的 token 降低 logits
- 或对高频 token 做惩罚

Mini-SGLang 目前的 `SamplingParams` 比较简化（更像教学/基线），之后扩展也很自然。

---

## 7. Mini-SGLang 的落点

采样入口在：

- `python/minisgl/engine/sample.py`

系统整体流程是：

1. engine forward 得到 logits
2. sampler 根据 `BatchSamplingArgs`（来自 `SamplingParams`）采样 next token
3. scheduler 把 next token 写入 `token_pool`，并发送给 detokenizer

如果你要理解“这些 token 是怎么被写入 token_pool / page_table 的”，回到：

- `docs/overview_zh.md` 的 Scheduler/KV cache 章节


