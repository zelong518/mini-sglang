# 04｜KV cache 基础：它是什么、为什么快、为什么占显存（面向推理小白）

> 目标：读完你能回答：
>
> - KV cache 到底缓存了什么？
> - 为什么它能让 decode 变快？
> - 为什么它会成为显存瓶颈？
> - Serving 系统一般怎么“管”KV cache？

---

## 1. KV cache 缓存的是哪两个东西？

在 Transformer 的 attention 里，每层会产生：

- Key：\(K\)
- Value：\(V\)

在自回归生成中：

- 历史 token 的 \(K,V\) 会被未来 token 反复使用
- 所以把它们缓存起来就叫 **KV cache**

因此 KV cache 的本质是：**按层存储“历史 token 的 K/V”**。

---

## 2. 为什么 KV cache 能显著加速 decode？

如果没有 KV cache，第 \(t\) 步生成时你要把 1..t 的序列整段再跑一遍，代价会随 \(t\) 增长。

有 KV cache 后：

- 历史 token 的 \(K,V\) 不用重算
- 只需要为新 token 计算新的 \(q_t, k_t, v_t\)
- 然后 \(q_t\) 直接和 cache 中的所有 \(K,V\) 做 attention

这样每步的“重算”被消除了，性能大幅提升。

---

## 3. 为什么 KV cache 会成为显存大头？

KV cache 会随序列长度线性增长，而且每层都要存：

- 大概规模（粗略）：
  - layers × seq_len × heads × head_dim × 2（K+V）× dtype_size

举个直观例子：seq_len 从 1k 增长到 32k，KV cache 直接扩大 32 倍。

因此在线 Serving 要么：

- 限制 max_seq_len / max_tokens
- 做 chunked prefill 控制峰值
- 做缓存复用（prefix cache）减少重复 KV
- 做 eviction（逐出）回收空间

---

## 4. KV cache 的“管理问题”是什么？

Serving 系统里同时会有很多请求，它们的 KV cache 占用需要：

- **分配**：新 token 的 KV 写到哪里？
- **复用**：如果新请求和旧请求有共享前缀，能否复用旧 KV？
- **回收**：请求结束后哪些 KV 可以保留（做缓存），哪些需要释放？
- **逐出**：显存不足时，优先丢掉谁？

这就是为什么系统会出现：

- naive cache（不复用，结束就释放）
- radix cache（复用前缀，保留历史 KV）
- LRU/时间戳/引用计数等策略

---

## 5. Mini-SGLang 中对应的落点

### 5.1 KV cache 存储本体

- `python/minisgl/kvcache/mha_pool.py`
  - `MHAKVCache` 持有一个大 tensor（按 layer/head/page/head_dim）
  - `store_kv(...)` 调用 kernel 把新的 K/V 写入 cache

### 5.2 KV cache 的写入内核

- `python/minisgl/kernel/store.py`：`store_cache(...)`

### 5.3 “管理策略”（复用/逐出/锁）

- `python/minisgl/kvcache/naive_manager.py`
- `python/minisgl/kvcache/radix_manager.py`

以及 scheduler 侧的页分配：

- `python/minisgl/scheduler/cache.py`：`CacheManager.allocate/free_and_cache_finished_req`

下一篇建议读（理解 paged KV/page table）：

- `05_paged_kv_and_page_table.md`


