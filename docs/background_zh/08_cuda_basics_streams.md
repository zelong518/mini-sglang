# 08｜CUDA 基础：device/stream/异步拷贝/事件（面向推理小白）

> 目标：你会理解推理系统里常见的这些词在说什么：
>
> - GPU device、CUDA stream
> - 异步拷贝（non_blocking）、pinned memory
> - CUDA event（同步点）
> - “GPU 等 CPU”到底是什么意思
>
> 这些概念会直接影响你读懂 Mini-SGLang 的 overlap scheduling、CPU↔GPU token 拷贝、以及 CUDA graph。

---

## 1. 一张 GPU 上的“并发”从哪里来？

很多人以为 GPU 只会“串行执行一个 kernel”。事实上在 CUDA 的抽象里：

- 一个 GPU device 可以同时管理多个 **CUDA streams**
- 每个 stream 内部是“顺序”的（按提交顺序执行）
- 不同 stream 之间在满足依赖关系的前提下可以并发（取决于硬件与资源）

推理系统经常用“多个 stream”来把：

- GPU 计算（模型 forward）
- 内存搬运（H2D/D2H 拷贝）
- 元数据准备（一些轻量 kernel）

尽量重叠起来，提高利用率。

---

## 2. CUDA stream 是什么（小白版）

你可以把 stream 想象成：

> **给 GPU 的一条“工作队列”**。

当你在 Python 里调用一个 CUDA op，本质上是“把任务塞进某个 stream 的队列里”，然后 GPU 在后台执行。

关键点：

- 如果你只用默认 stream（或同一个 stream），任务会排队串行
- 如果你拆成两个 stream，并且它们没有依赖，就有机会并发

---

## 3. pinned memory（页锁定内存）与 non_blocking 拷贝

### 3.1 CPU→GPU 的拷贝为什么经常是瓶颈？

推理系统需要在 CPU/GPU 之间搬运：

- token ids（一般是 int32）
- 一些元数据（cu_seqlens、positions…）
- 输出 token（GPU→CPU）

普通的 CPU 内存可能无法高效 DMA 到 GPU，因此 CUDA 提供 pinned memory：

> pinned memory 是“不会被操作系统换页”的 CPU 内存，GPU 可以更高效地从它 DMA。

### 3.2 `non_blocking=True` 代表什么？

在 PyTorch 里，很多 `.to(device, non_blocking=True)` 的含义是：

- 如果源 tensor 在 pinned memory 上
- 则可以发起异步拷贝（不阻塞当前 CPU 线程）

但注意：**non_blocking 不是魔法**：

- 如果源不是 pinned memory，可能还是会同步/慢
- 异步拷贝不代表“马上可用”，你仍然需要正确同步（用 event 或 stream sync）

---

## 4. CUDA event：你该在哪里同步？

一个常见模式是：

1. GPU 上算出 next tokens（tensor 在 GPU）
2. 发起 `to("cpu", non_blocking=True)` 拷贝
3. 记录一个 CUDA event（copy_done_event）
4. 之后在 CPU 侧需要读取结果之前，`event.synchronize()`

这比“每步都 `torch.cuda.synchronize()`”更精细，能减少不必要的全局同步。

---

## 5. “GPU 等 CPU”是什么？

如果你每步 decode 都要做很多 Python 工作（调度、构造 metadata、发消息），就会出现：

- CPU 没准备好下一步输入
- GPU 空转等待下一步 kernel 被提交

表现就是：

- GPU utilization 低
- tokens/s 上不去

解决办法（后面几篇会讲）：

- overlap scheduling（CPU 工作与 GPU 计算重叠）
- CUDA graph（减少每步提交 kernel 的 CPU 开销）

---

## 6. Mini-SGLang 里怎么体现这些概念？

### 6.1 双 stream overlap

在 `python/minisgl/scheduler/scheduler.py`：

- `self.stream = torch.cuda.Stream(...)`：scheduler 自己的 stream
- `self.engine.stream`：engine forward 的 stream
- `engine.stream.wait_stream(self.stream)`：建立依赖，确保准备工作先完成再跑 forward

### 6.2 pinned memory 的使用

你会看到很多类似：

- `pin_memory=True`
- `.to(device, non_blocking=True)`

用于把 token ids / cu_seqlens 等从 CPU 快速搬到 GPU。

### 6.3 event 同步

engine forward 输出里包含：

- `copy_done_event`（见 `python/minisgl/engine/engine.py` 的 `ForwardOutput`）

scheduler 在处理上一轮输出时，会 `copy_done_event.synchronize()` 再读取 CPU token。

下一篇建议读：

- `09_cuda_graph_basics.md`


