# 06｜Batching 与调度：为什么需要 Scheduler（面向推理小白）

> 目标：你会理解：
>
> - 什么是 batching，为什么它能提高吞吐
> - 为什么 batching 会伤害延迟
> - Scheduler 在在线推理中要解决哪些“冲突目标”
> - Mini-SGLang 的调度循环大致在做什么

---

## 1. 什么是 batching？

LLM 的一次 forward 可以把多条请求一起算（batch size = 多条请求数）。

直觉：

- GPU 擅长并行，batch 越大越能吃满 GPU
- 很多 kernel 的吞吐在更大 batch 下更高

因此 batching 通常能提高 **tokens/s**。

---

## 2. 为什么 batching 会伤害延迟？

在线系统里，用户希望尽快看到结果：

- 如果你一直等更多请求凑成大 batch，用户会“排队”

这就是 throughput 和 latency 的经典冲突：

- batch 大：吞吐高，但排队/TTFT 可能上升
- batch 小：TTFT 低，但 GPU 利用率可能低

调度器要在两者之间做权衡。

---

## 3. prefill batching 和 decode batching 是两回事

### 3.1 prefill batching

难点：

- prompt 变长：长度差异巨大（1k vs 32k）
- 显存峰值：prefill 一次性要写入大量 KV

常见策略：

- token budget（一次最多处理多少 token）
- chunked prefill（长 prompt 切块）

### 3.2 decode batching

特点：

- 每条请求每步只生成 1 个 token（形态更统一）
- 很适合把正在生成的请求集合在一起做 decode batch

常见策略：

- 把所有“还没结束”的请求放进 running set
- 每轮从 running set 选出一个 decode batch

---

## 4. Scheduler 在在线推理中要解决什么？

你可以把 Scheduler 的任务理解为五个问题：

1. **接入**：新请求来了，先放到哪里？
2. **准入控制**：显存/页不足时，这轮能接多少？
3. **组 batch**：这一轮跑 prefill 还是 decode？选哪些请求？
4. **准备 metadata**：把 input_ids/out_loc/page_table/attn_metadata 准备好
5. **收尾**：拿到 next token，决定是否结束，回收资源，发送结果

---

## 5. Mini-SGLang 的调度（你应该如何读）

### 5.1 关键数据结构

- pending（等待 prefill）：`PrefillManager.pending_list`
- running（等待 decode）：`DecodeManager.running_reqs`

### 5.2 “prefill 优先”的策略

Mini-SGLang 目前的策略是：

- 能跑 prefill 就先跑 prefill
- 否则跑 decode

这能降低新请求的 TTFT（让新请求尽快进入系统），但可能会牺牲部分 decode 吞吐。

### 5.3 overlap scheduling（把 CPU 工作埋到 GPU 时间里）

decode 很容易被 CPU 开销拖住，因此 Mini-SGLang 使用：

- 一个 stream 做调度/准备
- 一个 stream 做 engine forward
- 同时处理上一轮输出（pipeline）

---

## 6. 对应到源码（入口）

- `python/minisgl/scheduler/scheduler.py`
  - `_schedule_next_batch`：决定本轮跑 prefill 还是 decode
  - `_prepare_batch`：准备 out_loc/page_table/attn_metadata
  - `overlap_loop`：核心 pipeline

- `python/minisgl/scheduler/prefill.py`
  - `PrefillAdder`/`PrefillManager`：token budget + chunked prefill

- `python/minisgl/scheduler/decode.py`
  - `DecodeManager`：running set 的简单管理

如果你下一步要理解“为什么要 CUDA graph”，建议读：

- `09_cuda_graph_basics.md`


