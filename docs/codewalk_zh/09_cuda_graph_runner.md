# 09｜CUDA Graph：GraphRunner 捕获与回放（贴近源码的读法）

> 目标：让你读 `python/minisgl/engine/graph.py` 不再迷路：
>
> - capture 阶段到底在固定什么东西？
> - replay 阶段到底在更新什么东西？
> - 为什么要 padding batch size？
> - attention backend 为什么要实现 `prepare_for_capture/replay`？

---

## 1. 先打开哪些文件？

按这个顺序：

1. `python/minisgl/engine/graph.py`
2. `python/minisgl/engine/engine.py::Engine.forward_batch`（看 graph_runner 的使用点）
3. `python/minisgl/attention/base.py`（capture/replay hooks）
4. `python/minisgl/attention/fa3.py` 与 `python/minisgl/attention/fi.py`（各自如何配合 graph）

补课（如果你对 CUDA graph 概念不熟）：

- `docs/background_zh/09_cuda_graph_basics.md`

---

## 2. GraphRunner 在系统里的定位

GraphRunner 不是“替代模型 forward”，而是一个可选加速层：

- 满足条件时：replay graph（少 CPU launch）
- 不满足条件时：走普通 `model.forward()`

因此你读 `Engine.forward_batch` 时要先抓住这个分支。

---

## 3. capture：你到底在“捕获”什么？

Graph capture 的核心目标是固定：

- 一步 decode 的 kernel launch 序列
- 以及相关的 buffer shape（必须稳定）

因此 capture 时会准备：

- 固定形状的 `Batch`（dummy req 填充）
- 固定形状的 `input_ids/out_loc/page_table/metadata` buffer
- 固定形状的 logits buffer（最大 bs）

你会在 `graph.py::GraphRunner.__init__` 里看到：

- 为多个 batch size 捕获多个 `torch.cuda.CUDAGraph()`
- 以及一个共享的 memory pool（`pool = g.pool()`）用于减少重复分配

---

## 4. 为什么要 padding batch size？

因为每个 CUDAGraph 对应的是固定 shape。

如果你只捕获了 bs=8 的图，那么实际 bs=6 的 batch 需要：

- padding 到 8（添加 dummy req）
- 让 kernel shape 与 capture 保持一致

因此你会看到：

- `GraphRunner.pad_batch(batch)`：决定 padded_size
- scheduler 在 `_prepare_batch` 中对 out_loc 做 pad（dummy page）

---

## 5. replay：每步真正变化的是什么？

decode 每步变化的输入通常只有：

- 本轮的 `batch.input_ids`（新 token ids）
- 本轮的 `batch.out_loc`（新 KV 写入页）
- 本轮的 `attn_metadata`（例如 seqlens/page indices/positions 等）

而模型权重、buffer shape 都不变。

因此 replay 的结构通常是：

1. `attn_backend.prepare_for_replay(batch)`：
   - 把本轮的 input_ids/out_loc/metadata copy 到 capture buffer
2. `g.replay()`：
   - 执行 capture 时固定的 kernel 序列
3. 返回 logits（预分配 buffer 的切片）

---

## 6. 为什么 attention backend 必须参与 capture/replay？

因为 attention backend 持有/依赖的 metadata 很多，并且是 ragged/paged 的：

- capture 阶段需要把 wrapper/buffer 预先准备好
- replay 阶段需要把“本轮变化的那几个张量”写到 capture buffer

所以 backend 接口里才会有：

- `init_capture_graph`
- `prepare_for_capture`
- `prepare_for_replay`

你可以对比：

- `FA3Backend.prepare_for_replay`：会 copy `input_ids/out_loc/cu_seqlens_k/positions/page_table` 等
- `FlashInferBackend.prepare_for_capture`：会创建 graph wrapper 并复用 int workspace buffer

---

## 7. 建议断点/打印点（高收益）

- `GraphRunner.can_use_cuda_graph(batch)`：
  - 打印 `batch.phase/batch.size/batch.padded_size`
- `GraphRunner.pad_batch(batch)`：
  - 看 bs=6 最终 pad 到哪个捕获过的 bs
- `GraphRunner.replay(batch)`：
  - 看 `attn_backend.prepare_for_replay` 具体 copy 哪些张量

下一篇建议读：

- `10_tp_and_distributed.md`


