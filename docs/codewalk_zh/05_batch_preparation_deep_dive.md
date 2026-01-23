# 05｜Batch 准备超细节：indices、token_pool、page_table、out_loc 怎么连起来

> 这一篇只讲一件事：Scheduler 在 `_prepare_batch` 里到底做了什么，为什么它是整套系统的“核心接线板”。
>
> 小白补课：
> - `docs/background_zh/05_paged_kv_and_page_table.md`
> - `docs/background_zh/08_cuda_basics_streams.md`

---

## 1. 先打开哪些文件与函数？

请直接定位到这些符号（按顺序）：

1. `python/minisgl/scheduler/scheduler.py::Scheduler._prepare_batch`
2. `python/minisgl/scheduler/scheduler.py::_make_2d_indices`
3. `python/minisgl/scheduler/cache.py::CacheManager.allocate`
4. `python/minisgl/scheduler/table.py::TableManager`（看 token_pool 是什么）
5. `python/minisgl/engine/engine.py`（看 page_table 是怎么创建的）

---

## 2. 目标：把“请求列表”变成 Engine 能跑的一批输入

给定：

- `batch.reqs`：一组 `Req`
- 每条 req 有：
  - `table_idx`（它在 token_pool/page_table 的行）
  - `cached_len/device_len`（本轮要 extend 的范围）

Scheduler 要准备出：

- `batch.out_loc`：本轮新增 token 的 page ids（写 KV 用）
- `batch.input_ids`：本轮 forward 的输入 token ids（读 token_pool 用）
- `batch.attn_metadata`：attention backend 需要的元数据（cu_seqlens/page indices/positions…）
- `load_indices/write_indices`：两套 1D 索引，用来在 2D 表上做 gather/scatter

---

## 3. 第一件事：计算 `needed_size` 与分配 `out_loc`

### 3.1 `extend_len` 的意义（每条 req 本轮新增多少 token）

对每条 req：

- `extend_len = device_len - cached_len`

prefill/extend 时可能 >1；decode 时通常等于 1。

### 3.2 `needed_size`

batch 的总新增 token 数：

- `needed_size = sum(req.extend_len for req in batch.reqs)`

### 3.3 分配页：`out_loc`

调用：

- `batch.out_loc = cache_manager.allocate(needed_size)`

得到：

- `out_loc`：GPU 1D int32 tensor，长度 = `needed_size`

它存的是 page ids，也就是“本轮新增 token 的 KV 要写到哪些物理页”。

---

## 4. 第二件事：把 2D 切片压成 1D indices（关键技巧）

系统里有两张 2D 表：

- `token_pool`：存 token ids
- `page_table`：存 page ids

但为了高效，scheduler 用 `.view(-1)` 把它们当成 1D，然后用 1D indices 做 gather/scatter。

### 4.1 `_make_2d_indices`：输入 ranges，输出 1D indices

它接受一个 `table_2d`（必须 contiguous 2D）以及 ranges 列表：

- `ranges = [(entry, begin, end), ...]`

返回：

- `indices_host`（CPU pinned）→ 拷到 GPU 的 `indices`

你可以把它理解为：

> 把一堆 2D 坐标（行/列范围）转换成“底层 1D 内存索引”，便于一次性 gather/scatter。

### 4.2 `load_indices`：本轮要读的输入 token ids 在 token_pool 的位置

构造 ranges：

- 对每条 req：`(req.table_idx, req.cached_len, req.device_len)`

原因：

- 本轮 forward 的输入 token 是“新扩展出来的那一段”
- 已经 cached 的前缀不需要再次作为输入（prefill 命中 prefix 时尤其关键）

于是：

- `load_indices = _make_2d_indices(token_pool, ranges_for_load)`
- `batch.input_ids = token_pool.view(-1)[load_indices]`

### 4.3 `write_indices`：本轮要把 next token 写回 token_pool 的位置

每条真实 req 都会在 forward 后得到一个 next token，需要写到：

- `token_pool[table_idx, device_len]`

因此 ranges 是：

- `(req.table_idx, req.device_len, req.device_len + 1)`

于是：

- `write_indices = _make_2d_indices(token_pool, ranges_for_write)`
- `token_pool.view(-1)[write_indices] = next_tokens_gpu`

---

## 5. 第三件事：把 `out_loc` 写进 `page_table`（核心接线）

你会在 `_prepare_batch` 里看到类似这行：

- `page_table.view(-1)[load_indices] = batch.out_loc`

直觉解释（非常重要）：

- `load_indices` 指向的是“本轮新增 token 所在的 (table_idx, position)”集合
- `out_loc` 是这些 token 对应的 page ids
- 写进去后：
  - attention backend 通过 page_table 能知道每个 token 的 KV 物理页在哪里
  - KV 写入 kernel 也能按 out_loc 把 K/V 写进正确位置

如果你只理解一件事：

> **load_indices 把“逻辑 token 位置”与“物理 page id”绑在一起**。

---

## 6. 第四件事：准备 attention metadata

在写完 page_table 后，scheduler 会调用：

- `engine.attn_backend.prepare_metadata(batch)`

不同 backend 会准备不同的 metadata：

- FA3：更偏向 cu_seqlens + 本 batch 的局部 page_table（2D）
- FI：更偏向 ragged indices（1D 拼接）+ wrapper.plan 所需的 cpu/gpu 张量

你可以暂时不深究 metadata 的全部字段，但要明确：

- **metadata 的输入依赖于 page_table/token_pool/out_loc**
- 所以必须在写 page_table 后再 prepare

下一篇建议读：

- `06_engine_forward_and_sampling.md`


