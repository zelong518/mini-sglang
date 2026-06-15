# 并行策略：实现细节与性能基准

本文档记录 mini-sglang 中四种并行方式的**实现细节**、**测试命令**与**性能数据**：

| 方式 | 分支 | 基点 | 切分维度 |
|---|---|---|---|
| **TP** 张量并行 | `main` / 上游 | — | 注意力头 / MLP·专家中间维 |
| **EP** 专家并行 | `feat/ep-token-shard` | 上游 PR #96 (`60520ae`) | MoE 专家 |
| **PP** 流水线并行 | `feat/pp-parallel` | `db31896` | 解码器层 |
| **CP** 上下文并行 | `feat/cp-parallel` | `db31896` | KV 序列位置（decode） |

> 单一并行方式，不交叉组合（`tp_size` 与 `ep/pp/cp` 中至多一个 > 1）。

---

# 一、测试方法

- **硬件**：H200 (sm90)。基准 1 用 8 卡，基准 2 用 4 卡（GPU 0-1 当时被他人占用）。
- **权重**：`--dummy-weight`（随机权重）。FLOPs / 显存 / 带宽特性与真实权重一致，仅输出无意义
  ——基准只测速度，且避免反复加载 470 GB 权重。
- **负载**：单请求（并发 = 1），衡量单请求 / 低并发长上下文延迟。
- **测量**：**全部客户端测量**（流式时间戳）。服务端 `usage` 字段恒为 0，不可信。
  - `prefill 吞吐 = 输入token数 / TTFT`
  - `decode 吞吐 = (解码token数 − 1) / (末token时间 − 首token时间)`
- **防坑**：① 每请求加唯一随机前缀（防 radix 前缀缓存命中）；② 首请求预热丢弃；
  ③ 按运行唯一的 rng 种子（防跨运行命中缓存）。

## 测试脚本 `debug/bench_parallel.py`

```
python debug/bench_parallel.py <model_path> <url> <label> <lengths_csv> <decode_steps>
```

核心逻辑：
```python
# 1) 造定长 prompt：唯一随机前缀(64 tok) + 重复文本填充到目标长度
nonce = [rng.randint(1000, 150000) for _ in range(64)]
body  = (_BASE * (L // len(_BASE) + 1))[: L - 64]
prompt = tok.decode(nonce + body)

# 2) 预热一次（丢弃），消除一次性初始化开销
# 3) 对每个长度发 /generate (stream=True, ignore_eos=True)，记录每 token 时间戳
t0 = perf_counter()
r = requests.post(url+"/generate", json={"prompt":prompt,"max_tokens":D,"ignore_eos":True}, stream=True)
for line in r.iter_lines():
    if line.startswith(b"data:") and b"[DONE]" not in line:
        tics.append(perf_counter())
ttft        = tics[0] - t0                      # = prefill 时间
prefill_tps = L / ttft
decode_tps  = (len(tics)-1) / (tics[-1]-tics[0])
```

---

# 二、测试命令

### 启动服务（dummy 权重）

```bash
M=/volume/posttrain/models/Qwen3-235B-A22B   # 或 Qwen3-30B-A3B
BASE="python -m minisgl --model $M --dummy-weight --max-running-requests 4 --memory-ratio 0.9"

# TP=8   （baseline；--disable-pynccl 见“引擎边界”，>256k 必需）
$BASE --tp-size 8 --disable-pynccl --max-seq-len-override 600000 --port 1934

# EP=8   （feat/ep-token-shard 分支）
$BASE --tp-size 8 --ep-size 8 --disable-pynccl --max-seq-len-override 600000 --port 1936

# PP=8   （feat/pp-parallel 分支；显式定页绕开自动定页）
$BASE --pp-size 8 --num-pages 200000 --max-seq-len-override 150000 --port 1938

# CP=4   （feat/cp-parallel 分支）
$BASE --cp-size 4 --max-seq-len-override 600000 --port 1944

# 限定 GPU（基准 2 用 2-5 号卡）：前面加  CUDA_VISIBLE_DEVICES=2,3,4,5
# 对照：TP 关 CUDA graph 加  --graph 0
```

### 运行基准

```bash
python debug/bench_parallel.py $M http://127.0.0.1:1934 TP8 8000,32000,128000,256000,512000 32
```

---

# 三、测试结果

## 基准 1：Qwen3-235B-A22B，8×H200（`memory_ratio=0.9`，TP/EP 用 `--disable-pynccl`）

**Prefill 吞吐 (tok/s)**

| 输入 | TP=8 | EP=8 | PP=8 |
|---|---|---|---|
| 8k   | **27,497** | 9,241  | 6,026 |
| 32k  | **27,315** | 15,363 | 4,476 |
| 128k | **14,940** | 10,823 | 2,119 |
| 256k | **12,840** | 7,410  | — |
| 512k |  **4,689** | 4,542  | — |

**Decode 吞吐 (tok/s)**

| 输入 | TP=8 | EP=8 | PP=8 |
|---|---|---|---|
| 8k   | **77.5** | 8.2 | 50.3 |
| 32k  | **72.5** | 8.0 | 47.5 |
| 128k | **22.2** | 6.0 | 21.8 |
| 256k | **22.9** | 6.0 | — |
| 512k | **22.5** | 5.8 | — |

**TTFT (s)**

| 输入 | TP=8 | EP=8 | PP=8 |
|---|---|---|---|
| 8k   | **0.29** | 0.87 | 1.33 |
| 128k | **8.57** | 11.8 | 60.4 |
| 512k | **109.2** | 112.7 | — |

> PP 单请求 >128k 极慢（stage 串行、无流水线重叠），未测 256k/512k。

## 基准 2：Qwen3-30B-A3B，4×H200（四方对比 + TP 关 graph 对照）

**Prefill 吞吐 (tok/s)**

| 输入 | TP=4 | EP=4 | PP=4 | CP=4 |
|---|---|---|---|---|
| 8k   | **57,630** | 33,532 | 15,479 | 19,422 |
| 32k  | **55,573** | 41,457 | 23,561 | 22,310 |
| 128k | **29,814** | 26,408 |  9,530 |  9,037 |
| 256k | **17,874** | 16,776 |  5,226 |  4,994 |
| 512k |  **9,816** |  9,679 |  2,721 |  2,619 |

**Decode 吞吐 (tok/s)**

| 输入 | TP=4 默认 | TP=4 关graph | EP=4 | PP=4 | CP=4 |
|---|---|---|---|---|---|
| 8k   | **206** | 31.3 | 15.4 | 140  | 27.6 |
| 32k  | **186** |  —   | 15.5 | 138  | 27.5 |
| 128k | 48.9    | 21.1 | 12.3 | **46.0** | 21.7 |
| 256k | 48.9    | 22.9 | 12.1 | 44.5 | 20.2 |
| 512k | 48.5    |  —   | 12.0 | 45.2 | 21.2 |

**TTFT (s)**

| 输入 | TP=4 | EP=4 | PP=4 | CP=4 |
|---|---|---|---|---|
| 8k   | **0.14** | 0.24 | 0.52 | 0.41 |
| 128k | **4.29** | 4.85 | 13.4 | 14.2 |
| 512k | **52.2** | 52.9 | 188  | 195  |

## 结论

1. **Prefill**：TP/EP 把单请求并行到多卡 → 快；PP（stage 串行）/ CP（prefill 复制）退化为
   ~单卡 → 慢 ~3×。
2. **Decode 决定性因素是 CUDA graph**：TP 默认 206，关 graph 暴跌到 31.3（6.6×）。
   EP/PP/CP 因 decode 含集合通信被迫禁 graph。
3. **同为 eager 时**：PP(140) > TP(31) ≈ CP(28) > EP(15)。PP 全程仅 3 次 P2P；CP 每层
   all-gather+merge 的开销被"每 rank 只读 1/cp KV"抵消；EP 每 MoE 层 ≥2 次 all-to-all 最慢。
4. **最大优化点**：让 EP/PP/CP 的 NCCL 集合通信也能进 CUDA graph 捕获。

---

# 四、各分支实现细节

四种方式共用一套"world / 进程"模型：启动器 spawn `tp_size × (ep|pp|cp)_size` 个进程，每进程
一张卡跑一个 scheduler+engine。单模式下另一维 = 1。新增维度都沿用 TP 的 `DistributedInfo`
注册模式（`set_xx_info` / `get_xx_info`）和配置透传（`--xx-size` → `ServerArgs` → `EngineConfig`）。

## EP — 专家并行（`feat/ep-token-shard`，基于上游 PR #96）

**PR #96 已有部分**：每 rank 持 `num_experts/ep_size` 个完整专家；MoE forward 用
`all_to_all_single` 把 (token,expert) 对按目标 rank 派发 → 本地 `fused_experts_impl` → 再
all-to-all 收回 → 反排序加权。复用 `fused_topk`，无新核。

**我补充的核心修复（`cc6e46a`，`python/minisgl/moe/ep.py`）**：
- **问题**：激活在 TP 下是**复制**的（每 rank 持全量 token）。原 EP 让每 rank 都派发**全部**
  token → 拥有某组专家的 rank 收到 `ep_size` 份重复 → **每 rank 都算了整个 MoE 的量**（冗余
  `ep_size` 倍），比 TP 还慢。
- **修复**：在 MoE 入口把复制的 token **按 rank 切片**（每 rank 只取 `1/ep_size` 连续 token），
  只对本地切片做 dispatch/combine，结束后 `all_gather_into_tensor` 还原成全量复制激活。
  全局派发对数从 `ep_size·N·topk` 降回 `N·topk`，每 rank 只算 `1/ep_size`。
  ```python
  chunk = (num_tokens + ep_size - 1) // ep_size
  local = hidden_states[ep_rank*chunk : (ep_rank+1)*chunk]   # 切片零通信（本就复制）
  out_local = _dispatch_combine(local, ...)                   # 仅本地切片走 all-to-all
  padded[:n_local] = out_local
  dist.all_gather_into_tensor(gathered, padded, group=cp/ep_group)   # 还原全量
  return gathered[:num_tokens]
  ```
- **空切片不死锁**：decode 时 `N < ep_size` 会有 rank 拿到 0 token，但仍**无条件执行所有集合
  通信**（count 交换、两次 dispatch all-to-all、combine all-to-all），保持各 rank 同步。
- **附带修复（`663611c`）**：`--dummy-weight` 路径原来把打包的专家张量拆成 per-expert key，
  而 `load_state_dict` 按打包 key 查找不重堆 → `KeyError`。改为直接 randn 打包张量。

**验证**：与纯 TP 输出逐 token 一致；decode N=1 不死锁。

## PP — 流水线并行（`feat/pp-parallel`，基于 `db31896`）

按层把模型切成 `pp_size` 个 stage，每 stage 一张卡。**单模式下 `tp=1`**（stage 内不切头/权重）。

- **层切分**（`distributed/info.py::pp_layer_range`）：均衡连续切分，前 `num_layers % pp_size`
  个 stage 各多 1 层。
- **stage 间通信**（`distributed/impl.py`）：用 NCCL **P2P `send`/`recv`** 传"物化的残差流"
  hidden（`x + residual`）；下个 stage 以 `residual=None` 续上（`RMSNormFused(h, None)` 返回
  residual=h，正好接续）。
- **模型**（`models/qwen3.py` / `qwen3_moe.py`）：首 stage 做 embed，末 stage 做 norm+lm_head，
  中间 stage `recv → 跑本地层 → send`。绑定词嵌入（tied）下 PP 直接报错（embed 在首、lm_head
  在末，无法共享）。
- **采样**（`engine.py::forward_batch`）：只有末 stage 有 logits → 它采样后用 `pp_broadcast`
  把 next tokens 广播给所有 stage，使各 stage 的 token pool 一致。**scheduler 几乎不用改**
  （batch 经 gloo 组广播到所有 stage，各 stage 对称）。
- **KV cache**：每 stage 只为**本地层数**分配（`num_layers_override`）；层 id 用本地编号；
  权重加载器跳过非本 stage 的层、把全局层号**重映射为本地**（与 EP 跳专家同套路），embed 只在
  首 stage、norm/lm_head 只在末 stage。
- **两个为真实模型修的 bug**：① 不均匀切层时各 stage 算出的 `num_pages` 不同 → 对 `num_pages`
  做 **MIN all-reduce** 使各 stage 一致（页表共享）；② PP 各 stage 显存天然不均（层数差、
  embed/lm_head），**跳过 TP 的 2GB 显存均衡检查**。
- CUDA graph 禁用（decode 有 P2P）。

**验证**：稠密 Qwen3-8B-Base pp=4 与单卡逐 token 一致；MoE 30B pp=8、235B pp=8 可跑、连贯。

## CP — 上下文并行（`feat/cp-parallel`，基于 `db31896`）

**decode 时**按 KV 序列位置分片。**单模式下 `tp=1`**（全头、KV 不按头切）。

- **机制**（`attention/fi.py`）：每 rank 持**全量复制 KV**，但 decode 时只对自己的**连续
  `1/cp` KV 块**算 partial attention（flashinfer `return_lse=True`），再 `cp_all_gather`
  收齐各 rank 的 `(out, lse)`，用 `flashinfer.merge_state`（online-softmax）合并 —— **数值精确**。
  ```python
  # prepare_metadata: decode 时把每个请求的 KV 页切成 1/cp 连续块，本 rank 只取自己的块
  lo, hi = self._cp_slice(req.device_len)         # [rank*L//cp, (rank+1)*L//cp)
  indices = page_table[req.table_idx, lo:hi]
  # forward: 对本块算 partial + lse，再跨 rank 合并
  out, lse = wrapper.run(q, kv_cache, return_lse=True)
  outs, lses = cp_all_gather(out), cp_all_gather(lse)
  v, s = outs[0], lses[0]
  for r in range(1, cp_size):
      v, s = flashinfer.merge_state(v, s, outs[r], lses[r])
  return v
  ```
  decode 时 query 在所有 key 之后 → 无 causal 问题，merge 精确等价于全量 attention。
- **prefill 保持复制**（每 rank 跑全量），用 `fa` 后端；decode 用 `fi` 后端（CP merge 代码所在）
  → 注意力后端固定为 `fa,fi`。
- 因 decode 有 all-gather，CUDA graph 禁用。
- **当前局限**：prefill 复制、KV 复制（**无显存节省**）、禁 graph。要兑现长上下文价值需扩展为
  **sharded-KV 存储 + ring-attention prefill（causal 分块）+ graph 兼容的 decode 集合通信**。

**验证**：Qwen3-8B-Base cp=8 与单卡逐 token 一致（33 token 贪心），连贯，无死锁。

---

# 五、引擎稳定性边界与指标坑

**边界**：
- pynccl 自定义核 **>256k 上下文崩溃**（`CUDA illegal memory access`）→ 必须 `--disable-pynccl`。
- **~1M 崩溃**：疑为注意力索引 int32 溢出（1M×32×128 ≈ 4e9 > 2³¹）。235B 单请求可行上限
  ≈ **512k**（2M 在 8×H200 放不下，KV 需 ~770 GB）。

**已规避的指标坑**：
1. radix 前缀缓存命中 → prefill 假高（曾 37 万 tok/s）。修复：每请求唯一随机前缀。
2. 固定 rng 种子跨运行命中旧缓存。修复：按运行唯一种子。
3. 首请求 warmup 污染首个测量点。修复：预热丢弃。
4. 服务端 `usage` 恒为 0。修复：纯客户端测量。
