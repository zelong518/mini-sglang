# 08｜KV cache 与 Radix cache：页分配、复用、回收与逐出（生命周期导读）

> 读完这一篇，你应该能把下面这条链路串起来：
>
> - 本轮新增 token → `CacheManager.allocate` 分配页 → `out_loc`
> - attention forward 写 KV → KV cache buffer
> - 请求结束 → `free_and_cache_finished_req` → radix insert + unlock + free
> - 新请求到来 → `match_prefix` 命中 → lock handle → 复用前缀
> - 内存不够 → `evict` 回收可逐出页

---

## 1. 建议先打开哪些文件？

按这个顺序：

1. `python/minisgl/scheduler/cache.py`（CacheManager：free_slots + evict + insert）
2. `python/minisgl/kvcache/__init__.py`（create_cache_manager）
3. `python/minisgl/kvcache/naive_manager.py`（naive baseline）
4. `python/minisgl/kvcache/radix_manager.py`（radix tree + ref_count + evict）
5. `python/minisgl/kvcache/mha_pool.py`（KV 存储与 store_kv）
6. `python/minisgl/kernel/store.py`（store_cache kernel）
7. `python/minisgl/kernel/radix.py`（fast_compare_key）

---

## 2. 两层“缓存管理”：为什么既有 CacheManager 又有 RadixCacheManager？

你会在代码里看到两层：

- `scheduler/cache.py::CacheManager`
  - 管理“物理页”资源：free list + allocate + 与 cache manager 协作 evict

- `kvcache/radix_manager.py::RadixCacheManager`（或 naive）
  - 管理“前缀映射”：token 前缀 ↔ page ids
  - 管理“哪些前缀页可以逐出”：ref_count / timestamp / size_info

直觉：

- CacheManager 负责“页从哪里来/回哪里去”
- RadixCacheManager 负责“哪些页代表可复用的缓存前缀，以及是否可逐出”

---

## 3. allocate：页不够时发生什么？

文件：`scheduler/cache.py::CacheManager.allocate`

逻辑（直觉版）：

1. 如果 `free_slots` 足够：直接切一段返回
2. 否则：
   - 计算需要额外回收多少页
   - 调 `manager.evict(needed)` 回收一批可逐出页（page ids）
   - 把 free_slots + evicted 合并后再切一段返回

注意：如果 eviction 回收不够，会 assert 失败（说明缓存策略/容量不足）。

---

## 4. match：新请求如何命中前缀？

文件：`scheduler/cache.py::CacheManager.match_req`

它会把请求的 `input_ids` 送给底层 cache manager：

- radix：`match_prefix`
- naive：直接返回 prefix_len=0

radix 的匹配会返回：

- `handle`（带 prefix_len 与 node）
- `indices`（对应前缀的 page ids）

关键约束：

> `indices` 只有在 handle 被 lock 后才安全可用（否则可能被 evict）。

---

## 5. lock/unlock：ref_count 为什么能保护前缀不被逐出？

文件：`kvcache/radix_manager.py::RadixCacheManager.lock_handle`

直觉：

- lock：沿 node → root 的路径 ref_count++
  - ref_count 从 0→1 时，把节点的长度从 evictable_size 移到 protected_size

- unlock：沿路径 ref_count--
  - ref_count 从 1→0 时，把长度从 protected_size 移回 evictable_size

因此 eviction 只会回收：

- ref_count==0 的部分（evictable）

---

## 6. insert：请求结束后如何把 KV 变成可复用前缀？

文件：`scheduler/cache.py::CacheManager.free_and_cache_finished_req`

它做三件事（直觉版）：

1. `in_cache_len = manager.insert_prefix(input_ids, indices)`
   - 把 token ids→page ids 插入 radix tree
2. `self._free(indices[old_handle.cached_len : in_cache_len])`
   - 把“重复/无需保留”的那段页归还 free list
3. `self.unlock(old_handle)`
   - 解除 old_handle 对路径的保护引用

你可以把它理解为：

> 结束时把“有价值的前缀 KV”留在 radix cache 里，同时把不该占用的页释放掉。

---

## 7. evict：直觉上会淘汰谁？

文件：`kvcache/radix_manager.py::RadixCacheManager.evict`

你读这段时抓住三点：

- 只能 evict `ref_count==0` 的节点（否则会破坏正在运行的请求）
- 节点有 `timestamp`，访问会更新（更像 LRU 的一部分实现）
- eviction 会返回“被回收的 page ids”，交给 CacheManager 合并到 free_slots

---

## 8. KV 写入：page ids 最终怎么变成 K/V 的物理写入？

文件：`kvcache/mha_pool.py::MHAKVCache.store_kv`

关键点：

- `indices=out_loc` 是 page ids（int32）
- `store_cache` kernel 会把 k/v 写进 KV buffer 对应位置

你不需要先看懂 CUDA 内核细节，但要明确数据流：

> scheduler 分配页（out_loc）→ attention forward 写 KV（store_cache）→ 后续 attention 通过 page_table/indices 读取 KV。

下一篇建议读：

- `09_cuda_graph_runner.md`


