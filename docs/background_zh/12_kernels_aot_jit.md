# 12｜AOT/JIT 自定义内核：为什么推理系统需要 kernel（面向小白）

> 目标：理解：
>
> - 为什么 Python 写不动关键路径
> - 什么是 AOT（提前编译）与 JIT（运行时编译）
> - Mini-SGLang 的 tvm-ffi + csrc 是怎么工作的
> - `store_cache` / `indexing` / `fast_compare_key` 这些 kernel 分别在解决什么

---

## 1. 为什么推理系统要写 kernel？

推理系统的关键路径里，有两类操作非常敏感：

1. **高频小操作**（decode 每 token 都会发生）
   - 例如 KV 写入、索引 gather、前缀比较

2. **带宽/访存密集**（比算力更容易成为瓶颈）
   - 例如把 K/V 写到指定位置、按 indices 从大矩阵抓取行

如果用纯 Python 或通用算子拼装，往往会遇到：

- overhead 太大（每步 decode 都被 CPU launch/框架开销拖慢）
- 内存访问模式不理想（吞吐差）

因此系统会用 C++/CUDA 写一些“刚好够用”的 kernel，把性能抬起来。

---

## 2. AOT vs JIT：两种编译方式

### 2.1 AOT（ahead-of-time）

特点：

- 源码固定，编译出的模块也固定
- 适合逻辑不依赖运行时 shape/参数的场景

Mini-SGLang 例子：

- `fast_compare_key`（`radix.cpp`）：比较两段 token ids 的最长前缀
- `pynccl.cu`：NCCL wrapper（需要链接 `-lnccl`）

### 2.2 JIT（just-in-time / runtime specialization）

特点：

- 运行时根据一些参数（比如 element_size、线程数）生成专用实例
- 适合对 shape/对齐敏感、想做模板特化的 kernel

Mini-SGLang 例子：

- `store_cache`（`store.cu`）：根据 element_size 选择专用 kernel
- `indexing`（`index.cu`）：根据 element_size/num_splits 做专用 kernel

---

## 3. tvm-ffi 在这里扮演什么角色？

你可以把 tvm-ffi 当成：

> 一个轻量的 C++/CUDA 扩展编译 + Python 绑定工具。

Mini-SGLang 在 `python/minisgl/kernel/utils.py` 里封装了：

- `load_aot(...)`：编译 csrc/src 下的文件
- `load_jit(...)`：把 csrc/jit 下的文件 include 进来再编译（并可生成 wrapper）

这样你在 Python 里可以像调用函数一样调用 kernel：

- `module.launch(...)`
- 或 `module.fast_compare_key(...)`

---

## 4. Mini-SGLang 的几个关键 kernel 在干什么？

### 4.1 `store_cache`：把新 token 的 K/V 写入 paged KV cache

路径：

- `python/minisgl/kvcache/mha_pool.py` 的 `store_kv(...)`
  - 调用 `python/minisgl/kernel/store.py` 的 `store_cache(...)`

意义：

- decode 每步都要写 KV
- 写入位置由 `out_loc`（page ids）决定
- kernel 需要高效处理“按 indices scatter 写入”

### 4.2 `indexing`：从大矩阵里按 token ids 抽取行

典型用途：

- embedding lookup
- 或某些按 vocab range 的快速 gather

它在推理里常见且高频，因此用专用 kernel 提升吞吐。

### 4.3 `fast_compare_key`：Radix cache 的前缀比较加速

Radix cache 的关键操作是“比较两段 token ids 的最长公共前缀长度”。

如果用 Python 逐元素比，会拖慢调度器；因此提供 C++ AOT 实现。

---

## 5. 读源码时的抓手

当你看到：

- `load_aot(...)`：多半是“逻辑固定”的内核
- `load_jit(...)`：多半是“根据 element_size 等特化”的内核

看不懂内核细节也没关系，先抓住它们在系统链路中的角色：

- “在哪里被调用”
- “输入输出是什么”
- “为什么它会在关键路径出现”

然后回到 `docs/overview_zh.md` 对应章节看整体链路，会更容易消化。


