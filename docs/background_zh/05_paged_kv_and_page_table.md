# 05｜Paged KV & Page Table：用“页”管理 KV cache（面向推理小白）

> 目标：把下面这些词讲清楚，并能映射到 Mini-SGLang：
>
> - page / paged KV cache
> - page table（页表）
> - out_loc（本轮 KV 写入位置）
> - 为什么要“间接寻址”而不是把 KV 连续 append

---

## 1. 先从“直觉问题”开始：为什么不直接把 KV 按序列连续存？

如果只有一个请求，并且序列长度固定，把 KV 连续存当然最简单。

但在线 Serving 有这些现实问题：

- 同时很多请求，每个请求长度不同、增长速度不同
- 请求结束要回收空间
- 有些请求前缀可以复用（prefix cache），会出现“共享前缀 + 不同后缀”

如果你把 KV 存成“每个请求一段连续数组”，会遇到：

- **扩容/搬迁**：请求变长时可能要搬 KV（很贵）
- **碎片化**：回收后空洞多，难复用
- **共享困难**：前缀共享时更复杂

因此很多系统把 KV cache 做成“页式管理”。

---

## 2. Paged KV 的核心思想

把 KV cache 的存储空间切成很多固定大小的块（page）：

- 每个 page 由一个整数 `page_id` 标识
- 一个请求的序列（token positions）不必连续放在物理内存里
- 只要能从“(req, position)”映射到“page_id”，attention 就能找到对应 KV

这种做法像操作系统的虚拟内存：

- 逻辑地址（序列位置）→ 物理页（page id）

---

## 3. Page Table：从 token 位置映射到 page_id

可以把 page table 想成一个二维表：

- 行：请求（或请求在系统里的 slot）
- 列：token position（序列位置）
- 值：对应的 `page_id`

举个小例子（示意，不是实际代码）：

- req A 的第 0..3 个 token 存在 page [8, 9, 3, 20]
- req B 的第 0..3 个 token 存在 page [7, 1, 2, 2]（也可能重复/共享，取决于策略）

attention backend 只要拿到：

- 当前需要访问哪些 token positions
- 对应的 page table 行

就能定位到 KV cache 的物理地址。

---

## 4. out_loc：本轮“新 token KV”要写到哪里？

在 decode 的每一步，每个请求都会新增一个 token。

系统需要先分配物理 page（page_id），然后把新 token 的 K/V 写进去。

因此每轮会产生一个 `out_loc`：

- 它是一个 1D 列表（tensor），存的是**本轮新增 token**要写入的 page_id
- 之后 scheduler 会把 out_loc 写进 page table 的对应位置

---

## 5. Mini-SGLang 的落点（非常具体）

### 5.1 page_size=1 的意义

Mini-SGLang 里 `page_size=1`（见 `minisgl/core.py` 的 `Context` 和 `Engine` 初始化）意味着：

> **一个 token 对应一个 page_id**（教学友好，概念最直观）

真实工业系统常把 page_size 设成更大（比如 16/32 tokens），提升 locality、减少页表开销。

### 5.2 `page_table` 的创建

- `python/minisgl/engine/engine.py` 会创建 `page_table`（int32 CUDA tensor）
  - 形状大致是 `(max_running_req + 1, max_seq_len)`

### 5.3 `out_loc` 的分配与写入页表

- `python/minisgl/scheduler/cache.py`：`CacheManager.allocate(needed_len)` 分配 page ids
- `python/minisgl/scheduler/scheduler.py`：在 `_prepare_batch` 中
  - `batch.out_loc = cache_manager.allocate(...)`
  - 把 out_loc 写入 `page_table`（通过 indices 映射到对应行列）

### 5.4 KV 写入

- attention backend forward 时：
  - `kvcache.store_kv(k, v, batch.out_loc, layer_id)`
- 最终调用 `python/minisgl/kernel/store.py` 的 `store_cache(...)`

---

## 6. 你应该带着什么问题去看源码？

建议你盯住这条链路：

1. scheduler 选择 batch（prefill/decode）
2. 计算 `needed_size`（本轮新增 token 数）
3. `allocate` 得到 `out_loc`
4. 写入 `page_table`
5. attention backend 用 `page_table` 做 paged attention
6. 写入 KV cache

如果你能走通这条链路，paged KV 就算理解了。


