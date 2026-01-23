# 09｜CUDA Graph：为什么能让 decode 更快（面向推理小白）

> 目标：理解 CUDA graph 在推理系统里解决的“真正问题”是什么，以及它为什么更适合 decode 而不一定适合 prefill。

---

## 1. decode 的痛点：每 token 都要提交一堆 kernel

decode 的典型形态是：

- 每步只新增 1 个 token（batch 中每条请求也大多新增 1 个 token）
- 但是这一小步 forward 内部可能包含：
  - 若干 GEMM（线性层）
  - attention kernel
  - KV 写入 kernel
  - layernorm/activation 等

如果每一步都通过 Python 发起这些操作，会产生大量开销：

- Python 调度开销
- CUDA kernel launch 开销
- 元数据准备（indices/cuseqlens/page_table 等）

当模型不太大或 GPU 很快时，这些开销会变成系统瓶颈。

---

## 2. CUDA graph 在做什么？

CUDA graph 的核心是：

> 把“一段固定的 GPU 操作序列”捕获成一个图，之后重复执行时只需要 `replay()`。

对推理来说，通常希望捕获的是：

- decode 一步 forward 的 kernel 序列

这样每个 token：

- 不再需要重新发起一大堆 kernel launch
- 只需要更新少量输入 buffer（token ids、page ids、seqlens…），然后 replay

---

## 3. 为什么它更适合 decode 而不是 prefill？

CUDA graph 有一个天然前提：

- 你捕获的那段执行路径要“足够固定”

decode 的形态往往更固定：

- 每条请求 extend_len=1
- batch 的 shape 相对稳定
- 适合提前捕获多种 batch size 的图（bs=1/2/4/8/…）

prefill 的形态通常更不固定：

- 变长严重（prompt 长度差异巨大）
- chunked prefill 每次 extend_len 可能不同
- 图捕获与复用难度大

因此很多系统：**decode 用 CUDA graph，prefill 用高吞吐 kernel**。

---

## 4. 一个重要现实：batch size 需要“对齐/padding”

如果你捕获了 bs=8 的图，那么 replay 时通常也要满足相同的 shape。

系统一般做法：

- 选择一组可捕获的 batch size（比如 1/2/4/8/16）
- 实际 batch size 是 6 时，把它 padding 到 8（用 dummy req 填充）

这就是为什么你会在代码里看到：

- `padded_reqs`
- `pad_batch`

---

## 5. Mini-SGLang 的落点（你应该怎么读）

核心在 `python/minisgl/engine/graph.py` 的 `GraphRunner`：

- 启动时：
  - 预分配 logits buffer
  - 调用 attention backend 的 `init_capture_graph`
  - 逐个 batch size 捕获图（`torch.cuda.CUDAGraph()`）

- 运行时：
  - `can_use_cuda_graph(batch)`：decode 且 batch size 不超过 max_graph_bs
  - `pad_batch(batch)`：对齐到捕获过的 padded size
  - `prepare_for_replay(batch)`：把本步的 token ids/out_loc/metadata copy 到 capture buffer
  - `g.replay()`：执行图

一个关键点是：attention backend 需要配合 CUDA graph：

- `prepare_for_capture`
- `prepare_for_replay`

你会在 `python/minisgl/attention/fa3.py` 与 `fi.py` 看到这套接口。

---

## 6. 常见坑（你读代码时会遇到）

- **坑 1：捕获图会占显存**
  - graph replay 需要固定 buffer，往往会预留一部分显存

- **坑 2：图捕获时需要“热身”与同步**
  - 你会看到一些 synchronize/empty_cache/reset_peak_memory 等操作

- **坑 3：不是所有 batch 都能用 graph**
  - 超过 max_bs 或 shape 不一致时要走普通路径

下一篇建议读（理解 TP 与 NCCL）：

- `10_tensor_parallel_nccl.md`


