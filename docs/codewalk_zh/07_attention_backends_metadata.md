# 07｜Attention backends 与 metadata：FA3/FI 到底需要准备什么？

> 读完这一篇，你应该能回答：
>
> - 为什么需要 `prepare_metadata(batch)`？
> - FA3 与 FlashInfer 的 metadata 有什么差异？
> - 为什么 decode 更适合某些 backend + CUDA graph？

---

## 1. 先打开哪些文件？

按这个顺序：

1. `python/minisgl/attention/base.py`（接口：BaseAttnBackend/BaseAttnMetadata/HybridBackend）
2. `python/minisgl/attention/fa3.py`（FA3：FA3Metadata + prepare/replay）
3. `python/minisgl/attention/fi.py`（FI：FIMetadata + wrapper.plan/run）
4. `python/minisgl/scheduler/scheduler.py::Scheduler._prepare_batch`（在哪里调用 prepare_metadata）

---

## 2. 为什么 attention backend 一定要“先准备 metadata”？

Scheduler 把一批请求拼成 batch 后，attention kernel 需要知道：

- 每条请求当前的序列长度是多少（变长/ragged）
- 本轮 query 是哪些 token（prefill 是一段，decode 是 1）
- KV cache 的物理位置在哪里（paged KV：page ids / page_table）
- RoPE/位置编码需要的 positions

这些信息既不是模型权重的一部分，也不是一个简单的“输入 token ids”就能推出来的，
因此必须由 backend 在每轮（或每种形态）准备好。

---

## 3. 接口视角：`BaseAttnBackend`

在 `attention/base.py` 中，backend 需要实现：

- `prepare_metadata(batch)`
- `forward(q,k,v,layer_id,batch)`

以及 CUDA graph 相关的：

- `init_capture_graph(max_seq_len, bs_list)`
- `prepare_for_capture(batch)`
- `prepare_for_replay(batch)`

你可以把它理解为：

> backend 不仅是一段 kernel 调用，还负责把“本轮 batch 形态”翻译成 kernel 能吃的 metadata。

---

## 4. FA3（FlashAttention3）metadata：更像“局部 page_table + cu_seqlens”

文件：`attention/fa3.py`

FA3Metadata 里常见字段（直觉解释）：

- `cu_seqlens_k / cu_seqlens_q`：
  - ragged 序列的累计偏移（前缀和），让 kernel 知道每条样本在拼接布局里的起止位置
- `cache_seqlens`：
  - 每条样本当前有效长度（paged KV 的 key/value 长度）
- `positions`：
  - RoPE/位置编码所需位置（通常按 token 位置生成）
- `page_table`（注意：这里经常是“截取后的局部 2D”）：
  - 对每条 req 取 `page_table[table_idx, :max_seqlen_k]` stack 成 `(bs, max_seqlen_k)`

### 4.1 为什么要“局部 page_table”？

全局 page_table 是 `(max_running_req+1, max_seq_len)`，非常大。

对一个 batch 来说，kernel 只需要：

- 本 batch 的 bs 行
- 每行的有效长度（到 max_seqlen_k）

因此 backend 会把它裁剪成一个更小的 tensor，便于 kernel 访问。

---

## 5. FI（FlashInfer）metadata：更像“ragged indices + wrapper.plan”

文件：`attention/fi.py`

FIMetadata 的一个显著特点是：

- 同时保留 CPU 与 GPU 的一些张量（因为 flashinfer 的 `plan(...)` 接口要 CPU 侧 indptr/last_page_len 等）
- `indices` 往往是 1D 拼接的 page ids（ragged layout）

### 5.1 为什么会有 `wrapper.plan(...)`？

FlashInfer 的 wrapper 会对当前 batch 形态做一次“规划”：

- indptr/indices/last_page_len 等决定了 ragged KV 的结构

计划好之后：

- `wrapper.run(q, paged_kv_cache)` 会更快

### 5.2 decode 与 CUDA graph

FI 对 decode 有专门的 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`：

- 这能让 decode 的执行序列更固定、更适合捕获与回放

---

## 6. HybridBackend：prefill 和 decode 用不同后端

在 `attention/base.py` 的 `HybridBackend`：

- `batch.is_prefill` → prefill_backend
- `batch.is_decode` → decode_backend

这就是 `--attn fa3,fi` 的真实含义：

- prefill 用 FA3
- decode 用 FI

---

## 7. 建议断点/打印点（非常有效）

- `FA3Backend.prepare_metadata`：
  - 打印 `seqlens_q/seqlens_k/max_seqlen_q/max_seqlen_k`
  - 打印 `new_page_table.shape`

- `FlashInferBackend.prepare_metadata`：
  - 打印 `indices.numel()` 与 `cu_seqlens_k_cpu[-1]` 是否一致

- `BaseAttnBackend.forward` 调用点：
  - 观察每层都在写 KV（`store_kv`）

下一篇建议读：

- `08_kvcache_and_radix_lifecycle.md`


