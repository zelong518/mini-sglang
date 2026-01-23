# 04｜Scheduler 主循环：prefill/decode/overlap 的骨架怎么跑

> 读完这一篇，你应该能：
>
> - 把 scheduler 的主循环拆成“收消息→选 batch→准备→forward→处理输出→回收/回传”
> - 理解为什么会有 normal_loop 和 overlap_loop 两套路径
> - 知道“prefill 优先”策略在哪里实现

---

## 1. 先打开哪些文件？

按这个顺序：

1. `python/minisgl/scheduler/scheduler.py`
2. `python/minisgl/scheduler/prefill.py`
3. `python/minisgl/scheduler/decode.py`
4. `python/minisgl/scheduler/io.py`
5. `python/minisgl/engine/engine.py`（只看 `forward_batch` 接口）

---

## 2. Scheduler 初始化：它创建了哪些关键组件？

文件：`python/minisgl/scheduler/scheduler.py`（`Scheduler.__init__`）

你会看到它初始化了：

- `self.engine = Engine(config)`：每个 rank 一个 engine（绑定 `cuda:{rank}`）
- I/O mixin：`SchedulerIOMixin`（决定 rank0 如何收/发消息）
- 两个 CUDA stream：
  - `self.stream`：scheduler 的 stream（准备 indices/metadata）
  - `self.engine.stream`：engine forward 的 stream
- 三个 manager：
  - `PrefillManager`（pending 队列）
  - `DecodeManager`（running set）
  - `CacheManager`/`TableManager`（页分配 + token_pool 管理）

---

## 3. normal_loop vs overlap_loop：差别是什么？

### 3.1 normal_loop（不开 overlap）

当 `ENV.DISABLE_OVERLAP_SCHEDULING=1`：

- 每轮：收消息 → schedule batch → forward → 处理输出
- 没有“上一轮/下一轮”交错

优点：逻辑简单  
缺点：CPU 开销更容易暴露，decode 吞吐可能低

### 3.2 overlap_loop（默认）

overlap_loop 的结构是 pipeline：

- 这一轮在 engine stream 上跑 forward 的同时
- scheduler 在 CPU 侧处理上一轮输出、释放资源、发送 detokenize

关键变量是 `last_data`：

- `last_data = (ForwardInput, ForwardOutput)`
- 这一轮开始时先处理上一轮的 `last_data`

你可以把它理解为一个 2-stage pipeline：

- Stage A：准备 + 发起 forward
- Stage B：处理上一轮输出 + 回收 + 回传

---

## 4. 收消息：`receive_msg` 是怎么工作的？

I/O 逻辑在 `python/minisgl/scheduler/io.py`。

- 单卡：rank0 直接从 tokenizer 的 PULL 收消息
- 多卡：
  - rank0 从 tokenizer 收 raw bytes，再 PUB 广播给其它 ranks
  - 用 CPU 进程组 broadcast “本轮消息条数”

因此在 scheduler 主循环里：

- `for msg in self.receive_msg(blocking=...)` 这行是“控制面入口”

---

## 5. 处理消息：`_process_one_msg`

核心是处理 `UserMsg`：

- 长度校验（不能超过 max_seq_len）
- 修正 `max_tokens`（避免越界）
- `prefill_manager.add_one_req(msg)` 入队

还有一个 `ExitMsg` 用于退出。

---

## 6. 选 batch：`_schedule_next_batch`（prefill 优先）

你会看到当前策略是：

- `prefill_manager.schedule_next_batch(prefill_budget) or decode_manager.schedule_next_batch()`

所以是“prefill 优先，decode 兜底”。

这能降低新请求的 TTFT，但可能影响 decode 吞吐；未来可以扩展更多 policy。

---

## 7. forward 前准备：`_prepare_batch`

这一块是整个系统的核心（也最容易把小白搞晕）。

你先记住它做三件事：

1. 分配本轮 KV 写入位置：`batch.out_loc = cache_manager.allocate(...)`
2. 构造 indices：
   - `load_indices`（从 token_pool gather 本轮输入）
   - `write_indices`（把 next token 写回 token_pool）
3. 写 `page_table` 并准备 attention metadata：
   - `page_table.view(-1)[load_indices] = batch.out_loc`
   - `engine.attn_backend.prepare_metadata(batch)`

这块的超细解释在下一篇：

- `05_batch_preparation_deep_dive.md`

---

## 8. forward：`_forward`

核心动作：

- `self._load_token_ids`：`batch.input_ids = token_pool.view(-1)[load_indices]`
- `forward_output = engine.forward_batch(batch, sample_args)`
- `self._write_token_ids`：写回 next token（GPU）
- `decode_manager.add_reqs(batch.reqs)`：把可继续 decode 的 req 加入 running set

---

## 9. 处理输出与回收：`_process_last_data`

这一段做了：

- 等待上一轮 `copy_done_event`（只同步必要的 token copy）
- 对每条 req：
  - append host token
  - 判断 finished（max_tokens/EOS/越界）
  - rank0 发送 `DetokenizeMsg`
- 对 finished req：
  - 释放 `table_idx`
  - 把 prefix 插入 cache（radix/naive），归还页，unlock handle

Radix/KV 生命周期详解见：

- `08_kvcache_and_radix_lifecycle.md`


