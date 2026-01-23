# 06｜Engine forward + 采样：一次 forward 到底发生了什么？

> 读完这一篇，你应该能把“scheduler 准备完 batch 之后”的事情串起来：
>
> - Context 是怎么把 batch 暴露给模型层/attention 的
> - 为什么有 CUDA graph replay 分支
> - next token 是怎么采样出来并写回系统的
> - CPU/GPU 同步点（event）在哪里

---

## 1. 先打开哪些文件？

按这个顺序：

1. `python/minisgl/engine/engine.py::Engine.forward_batch`
2. `python/minisgl/core.py::Context.forward_batch`
3. `python/minisgl/engine/sample.py`（Sampler）
4. `python/minisgl/engine/graph.py`（GraphRunner：can_use/replay/pad）

---

## 2. Engine 的职责：单 rank 的“执行引擎”

你可以把 Engine 当成：

> 在一个 GPU 上跑模型的一切：加载模型、初始化 KV cache/attention、执行 forward、采样、CUDA graph 管理。

Scheduler 负责“什么时候跑/跑哪些 req/怎么准备输入”，Engine 负责“把这批输入真正算出来”。

---

## 3. `Context.forward_batch`：为什么要有全局 context？

模型 forward 的过程中（尤其 attention 层）需要访问：

- 当前 batch 的 phase（prefill/decode）
- 当前 batch 的 input_ids
- page_table/out_loc/attn_metadata 等

Mini-SGLang 用 `Context` 把当前 batch 设置为“全局可访问”：

- `with ctx.forward_batch(batch): ...`

这样 attention backend 与一些层可以通过 `get_global_ctx().batch` 取到当前 batch 的信息。

---

## 4. CUDA graph replay 分支：什么时候走？

在 `Engine.forward_batch` 中你会看到：

- `if graph_runner.can_use_cuda_graph(batch): logits = graph_runner.replay(batch)`
- `else: logits = model.forward()`

直觉：

- decode 更固定、更适合 replay
- prefill 变长严重，多半走普通 forward

如果你不理解 CUDA graph，先回补课：

- `docs/background_zh/09_cuda_graph_basics.md`

---

## 5. KV 写入发生在哪里？

注意：Engine.forward_batch 本身并不直接“写 KV”。

KV 写入发生在 attention backend 的 forward 里：

- `attn_backend.forward(...)` 内部会调用 `kvcache.store_kv(k, v, out_loc, layer_id)`

这也是为什么 scheduler 必须先准备好：

- `batch.out_loc`
- `page_table`（以及 metadata）

---

## 6. `req.complete_one()`：为什么 forward 后要更新 req 状态？

每次 forward（尤其 decode）会生成 1 个 token。

Engine 在 forward 后会对每条 req 调：

- `req.complete_one()`

它本质上会：

- 把 `cached_len` 推进到 `device_len`
- 再把 `device_len += 1`（为 next token 预留位置）

这一步很关键：它会影响下一轮的 `extend_len`、以及 token_pool/page_table 的写入位置。

---

## 7. 采样：logits → next token

采样发生在：

- `Sampler.sample(logits[:batch.size], args)` → `next_tokens_gpu`

并会强制成 int32（便于 token id 的统一处理）。

如果你想理解 temperature/top-k，补课：

- `docs/background_zh/07_sampling_basics.md`

---

## 8. GPU→CPU：为什么要 `copy_done_event`？

Engine 会做：

- `next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)`
- `copy_done_event = torch.cuda.Event(); copy_done_event.record()`

Scheduler 在处理上一轮输出时会：

- `copy_done_event.synchronize()` 后再读取 `next_tokens_cpu`

这是一个很典型的“细粒度同步”模式：

- 不用每步全局 `torch.cuda.synchronize()`
- 只在需要读取 CPU token 时同步

补课：

- `docs/background_zh/08_cuda_basics_streams.md`

---

## 9. 这一轮的输出最终如何回到用户？

链路是：

1. scheduler 把 `next_tokens_gpu` 写回 `token_pool`
2. rank0 把 `next_tokens_cpu` 包成 `DetokenizeMsg` 发给 detokenizer
3. detokenizer 生成增量字符串，发 `UserReply` 给 API
4. API streaming 返回客户端

下一篇建议读：

- `07_attention_backends_metadata.md`


