"""单步调试入口：在当前进程内（offline 单进程模式）跑完整的
tokenize -> schedule -> engine forward -> attention -> sample -> detokenize 流程。

为什么用这个而不是 `python -m minisgl`：
    `python -m minisgl` 是多进程架构（spawn 出 scheduler / tokenizer 子进程 +
    API server 主进程），普通断点打不进子进程。而这里的 `LLM` 直接继承 `Scheduler`、
    `offline_mode=True`，所有逻辑都在**当前进程同步执行**，断点可以随便打。

建议配合的环境变量（已在 .vscode/launch.json 里设好，命令行也可手动加）：
    MINISGL_DISABLE_OVERLAP_SCHEDULING=1  # 关闭多 stream 重叠调度，执行变线性、好跟断点
    CUDA_LAUNCH_BLOCKING=1                # CUDA 同步执行，报错栈指向真正出错的 kernel/行
    LOG_LEVEL=DEBUG                       # 打开 logger.debug / debug_rank0 日志

常用断点位置（你最近在读的代码）：
    engine/engine.py      Engine.__init__ / forward_batch / _determine_num_pages
    scheduler/scheduler.py  Scheduler._schedule_next_batch / _forward / _process_last_data
    scheduler/cache.py    CacheManager.allocate_paged / _allocate
    attention/fi.py       FlashInferBackend.prepare_metadata / forward
    engine/graph.py       GraphRunner._capture_graphs / replay
"""

from __future__ import annotations

import os

# 默认把调试友好的开关打开（若外部已设置则尊重外部值）。
os.environ.setdefault("MINISGL_DISABLE_OVERLAP_SCHEDULING", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("LOG_LEVEL", "DEBUG")

# 远程 attach 调试（CPU 机器 <-> GPU 机器）：
#   在 GPU 机器上设置 MINISGL_DEBUGPY=1（可选 MINISGL_DEBUGPY_PORT，默认 5678）后运行本脚本，
#   程序会监听该端口并等待 CPU 机器的 VS Code attach（见 launch.json 的
#   "debug: attach to GPU (debugpy)" 配置）。两台机器挂载同一共享目录、路径一致，
#   所以断点行号天然对应，pathMappings 用 localRoot == remoteRoot 即可。
if os.environ.get("MINISGL_DEBUGPY", "0").lower() in ("1", "true", "yes"):
    # 关掉 frozen modules 校验，避免 debugpy 警告 "may make the debugger miss breakpoints"。
    os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
    import debugpy

    _port = int(os.environ.get("MINISGL_DEBUGPY_PORT", "5678"))
    debugpy.listen(("0.0.0.0", _port))
    print(f"[debug] waiting for debugger attach on port {_port} ...", flush=True)
    debugpy.wait_for_client()
    print("[debug] debugger attached.", flush=True)

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM

# ============================ 可调参数 ============================
MODEL_PATH = "/volume/posttrain/models/Qwen3-0.6B"

# use_dummy_weight=True 用随机权重跳过真实权重加载，启动快、适合调结构性逻辑；
# 想看真实生成结果就改成 False。
USE_DUMMY_WEIGHT = False

# cuda_graph_max_bs=0 会禁用 CUDA graph（见 graph._determine_cuda_graph_bs：
# max_bs < 1 时返回 []），模型走 eager 模式，断点能进每一层、报错栈清晰。
# 想调试 CUDA graph 的 capture/replay 本身时，把它改成 None（自动）或具体数值。
DISABLE_CUDA_GRAPH = True

PROMPTS = [
    "用一句话介绍你自己。",
    "1 + 1 等于几？",
]
# =================================================================


def main() -> None:
    kwargs = {"use_dummy_weight": USE_DUMMY_WEIGHT}
    if DISABLE_CUDA_GRAPH:
        kwargs["cuda_graph_max_bs"] = 0

    print(f"[debug] building LLM, model={MODEL_PATH}, kwargs={kwargs}")
    llm = LLM(model_path=MODEL_PATH, dtype=torch.bfloat16, **kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,   # greedy，结果可复现
        max_tokens=2,     # 调试时少生成几个 token，跑得快
    )

    print(f"[debug] generating for {len(PROMPTS)} prompts ...")
    # 想跟踪一次完整 forward：在这一行设断点，step into generate -> run_forever ->
    # overlap_loop/normal_loop -> _schedule_next_batch / _forward。
    results = llm.generate(PROMPTS, sampling_params)

    for prompt, res in zip(PROMPTS, results):
        print("=" * 60)
        print(f"[prompt] {prompt!r}")
        print(f"[output] {res['text']!r}")
        print(f"[token_ids] {res['token_ids']}")

    llm.shutdown()


if __name__ == "__main__":
    main()
