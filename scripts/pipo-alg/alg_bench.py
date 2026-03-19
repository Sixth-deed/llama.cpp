import argparse
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass

# ================= Progress Bar Globals & Interface =================
_PB_CURRENT = 0
_PB_TOTAL = 0
_PB_WIDTH = 40
_PB_INITIALIZED = False

def _pb_render():
    """内部函数：渲染进度条到 stderr"""
    if not _PB_INITIALIZED:
        return

    # 计算百分比和填充长度
    percent = 0
    filled_len = 0
    if _PB_TOTAL > 0:
        # 防止超过 100%
        curr = min(_PB_CURRENT, _PB_TOTAL)
        percent = int(100 * curr / _PB_TOTAL)
        filled_len = int(_PB_WIDTH * curr / _PB_TOTAL)

    bar = '#' * filled_len + '-' * (_PB_WIDTH - filled_len)
    # 使用 \r 覆盖当前行，输出到 stderr 以免干扰日志文件
    sys.stderr.write(f"\r[{bar}] {percent:3d}%")
    sys.stderr.flush()

def pb_init(total: int):
    """初始化进度条"""
    global _PB_TOTAL, _PB_CURRENT, _PB_INITIALIZED
    _PB_TOTAL = total
    _PB_CURRENT = 0
    _PB_INITIALIZED = True
    # 初始打印空进度条
    _pb_render()

def pb_tick():
    """进度条前进一格（每个子进程运行结束后调用）"""
    global _PB_CURRENT
    if not _PB_INITIALIZED:
        return
    _PB_CURRENT += 1
    _pb_render()

def pb_adjust_total(delta: int):
    """动态调整总任务数（当某些子进程被跳过时调用）"""
    global _PB_TOTAL
    if not _PB_INITIALIZED:
        return
    _PB_TOTAL += delta
    # 调整后重新渲染以更新百分比
    _pb_render()

def pb_finish():
    """结束进度条，确保显示 100% 并换行"""
    global _PB_CURRENT, _PB_TOTAL
    if not _PB_INITIALIZED:
        return
    _PB_CURRENT = _PB_TOTAL
    _pb_render()
    sys.stderr.write("\n")
    sys.stderr.flush()
# ====================================================================

@dataclass
class BenchConfig:
    prefill_batch: int
    decode_len: int
    target_mem_usage: int
    model_path: Path
    # bench_output_dir 和 log_dir 将在 __post_init__ 中初始化

    def __post_init__(self):
        # 使用 f-string 格式化路径
        self.bench_output_dir = OUTPUT_DIR / f"p{self.prefill_batch}_d{self.decode_len}_m{self.target_mem_usage}_{self.model_path.name.rstrip('.gguf')}"
        self.bench_output_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.bench_output_dir / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)


SCRIPT_DIR = Path(__file__).resolve().parent
LLAMA_DIR = SCRIPT_DIR / "../../"

LLAMA_DIR = LLAMA_DIR.resolve()

PERF_BIN = LLAMA_DIR / "build-release" / "bin" / "test-backend-ops-perf2"
ALG_BIN = LLAMA_DIR / "build-release" / "bin" / "pipo-alg"
BENCH_BIN = LLAMA_DIR / "build-release" / "bin" / "pipo-alg-bench"

MODEL_QWEN_14B_Q4 = LLAMA_DIR / "../model/Qwen3-14B-Q4_K_M.gguf"
MODEL_QWEN_MOE_30B_Q4 = LLAMA_DIR / "../model/Qwen3-30B-A3B-Q4_K_M.gguf"

OUTPUT_DIR = LLAMA_DIR / "logs/alg_bench"

# 全局配置，将在 main 中根据 argparse 更新
refresh_perf_result = False
n_runs = 5




# ==========================
# Single Bench run config

# map from alg to alg-no
alg_map = {
    "dp": 0,
    "pf": 1,
    "static": 2
,
    "dp-2": 3,
    "dp-3":4,
}
test_cases = [
    BenchConfig(99, 32, 6500, MODEL_QWEN_14B_Q4),
    BenchConfig(99, 32, 4000, MODEL_QWEN_14B_Q4),
    BenchConfig(412, 100, 6500, MODEL_QWEN_14B_Q4),
    BenchConfig(4000, 96, 5500, MODEL_QWEN_14B_Q4),
    BenchConfig(99, 32, 6000, MODEL_QWEN_MOE_30B_Q4),
    BenchConfig(4000, 96, 5500, MODEL_QWEN_MOE_30B_Q4)
]
# map from (model, mem_usage) to ngl
base_map = {
    (MODEL_QWEN_14B_Q4, 6500): 32,
    (MODEL_QWEN_14B_Q4, 4000): 19,
    (MODEL_QWEN_14B_Q4, 5500): 27,
    (MODEL_QWEN_MOE_30B_Q4, 6000): 17,
    (MODEL_QWEN_MOE_30B_Q4, 5500): 16,
}

# =======================

log_file = None
def log(msg: str):
    global log_file
    if (log_file) :
        print(msg, file= log_file)


def run_command(cmd: list, stdout_path: Path, stderr_path: Path, check: bool = False) -> int:
    """
    通用函数：运行子进程并重定向输出到文件
    返回 returncode
    """
    # 确保输出文件的父目录存在
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"\nruning: {' '.join(cmd)}")
    try:
        with open(stdout_path, 'w') as out_f, open(stderr_path, 'w') as err_f:
            # 使用 subprocess.run 运行命令
            # text=True 表示以文本模式处理输入输出
            result = subprocess.run(cmd, stdout=out_f, stderr=err_f, text=True, check=check)
            return result.returncode
    except subprocess.CalledProcessError as e:
        return e.returncode
    except Exception as e:
        log(f"Error running command {' '.join(cmd)}: {e}")
        return -1
    finally:
        # 子进程运行结束（无论成功失败），进度条前进一格
        pb_tick()

def run_alg(op_perf_json: Path, cfg: BenchConfig, alg_name: str):
    """
    运行算法生成二进制 (pipo-alg)
    返回 0 表示成功，1 表示失败
    """
    alg_id = alg_map.get(alg_name)
    if alg_id is None:
        log(f"Unknown algorithm: {alg_name}")
        return 1

    # 确保目录存在
    alg_log_dir = cfg.log_dir / "algs"
    alg_result_dir = cfg.bench_output_dir / "algs"
    alg_log_dir.mkdir(parents=True, exist_ok=True)
    alg_result_dir.mkdir(parents=True, exist_ok=True)

    alg_log_path = alg_log_dir / f"alg_{alg_name}.log"
    alg_result_path = alg_result_dir / f"alg_{alg_name}.json"

    # 构建命令
    # Usage: ./pipo-alg -m <model_file> -perf <op_perf_json> -alg-no <int> -mem <int> ...
    cmd = [
        str(ALG_BIN),
        "-m", str(cfg.model_path),
        "-perf", str(op_perf_json),
        "-alg-no", str(alg_id),
        "-mem", str(cfg.target_mem_usage),
        "-max-batch", str(cfg.prefill_batch)
    ]

    # 运行命令
    ret = run_command(cmd, alg_result_path, alg_log_path, check=False)

    if ret != 0:
        log(f"Alg {alg_name} generation failed. Check {alg_log_path}")
        return 1

    return 0



def run_bench(cfg: BenchConfig, alg_name: str):
    """
    运行基准测试二进制 (pipo-alg-bench)
    """
    bench_log_dir = cfg.log_dir / "bench"
    bench_result_dir = cfg.bench_output_dir / "bench"
    bench_log_dir.mkdir(parents=True, exist_ok=True)
    bench_result_dir.mkdir(parents=True, exist_ok=True)

    bench_log_path = cfg.log_dir / "bench.log"
    log_path = bench_log_dir / f"{alg_name}.log"
    result_path = bench_result_dir / f"{alg_name}.json"

    alg_result_path = cfg.bench_output_dir / "algs" / f"alg_{alg_name}.json"

    args = [str(BENCH_BIN), "-m", str(cfg.model_path)]

    if alg_name == "base":
        # 获取 ngl，如果没有配置则默认 10
        ngl = base_map.get((cfg.model_path, cfg.target_mem_usage), 10)
        args += ["-ngl", str(ngl)]
    else:
        if not alg_result_path.exists():
            msg = f"Alg result file not found: {alg_result_path}, skipping bench for {alg_name}"
            log(msg)
            with open(bench_log_path, 'a') as f:
                f.write(msg + "\n")
            return

        args += ["-pipo", str(alg_result_path)]

    args += [
        "-n", str(cfg.decode_len),
        "-p", str(cfg.prefill_batch),
        "-run", str(n_runs)
    ]

    # 运行命令
    ret = run_command(args, result_path, log_path, check=False)

    if ret != 0:
        fail_msg = f"failed to bench alg {alg_name} for Bench Config {cfg}, more information in {log_path}"
        log(fail_msg)
        with open(bench_log_path, 'a') as f:
            f.write(fail_msg + "\n")

def bench_test_case(cfg: BenchConfig):
    global log_file

    bench_log_path = cfg.log_dir / "bench.log"

    log_file = open(bench_log_path, 'w')

    perf_result_path = cfg.bench_output_dir / "op_perf.json"
    perf_log_path = cfg.log_dir / "op_perf.log"

    # 1. 运行 Perf (如果未复用且文件不存在)
    if refresh_perf_result or not perf_result_path.exists():
        # Usage: test-backend-ops-perf2 -m <model> [-p prefill-batch-size] [-n n_decode]
        cmd = [
            str(PERF_BIN),
            "-m", str(cfg.model_path),
            "-p", str(cfg.prefill_batch),
            "-n", str(cfg.decode_len)
        ]

        ret = run_command(cmd, perf_result_path, perf_log_path, check=False)

        if ret != 0 or not perf_result_path.exists():
            fail_msg = f"failed to perf ops for Bench Config {cfg}, more information in {perf_log_path}"
            log(fail_msg)
            with open(bench_log_path, 'a') as f:
                f.write(fail_msg + "\n")
            pb_adjust_total(-1 - len(alg_map) * 2)
            return
    else:
        # Perf 被跳过，需要调整进度条总数，以保持百分比准确
        pb_adjust_total(-1)

    # 2. 运行 Base 基准测试
    run_bench(cfg, "base")

    # 3. 运行算法基准测试
    for alg_name, _ in alg_map.items():
        # 先生成算法配置
        ret = run_alg(perf_result_path, cfg, alg_name)

        if  ret == 0:
            run_bench(cfg, alg_name)
        else:
            log(f"Skipping bench for {alg_name} due to alg generation failure.")
            # Alg 生成失败，对应的 Bench 也会被跳过，需要调整进度条总数
            pb_adjust_total(-1)

    if log_file:
        log_file.close()

def main():
    global OUTPUT_DIR, refresh_perf_result, n_runs

    parser = argparse.ArgumentParser(description="Benchmark PIPO Algorithms")
    parser.add_argument('-o', '--output', type=Path, default=OUTPUT_DIR, help="Output directory for logs and results")
    parser.add_argument('--refresh-perf-result', action='store_true', help="Refresh existing performance result JSON if exists")
    parser.add_argument('--run', type=int, default=5, help="Number of runs for benchmarking")

    args = parser.parse_args()

    # 更新全局配置
    OUTPUT_DIR = args.output
    refresh_perf_result = args.refresh_perf_result
    n_runs = args.run

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    total_steps = len(test_cases) * (1 + 1 + len(alg_map) * 2)
    pb_init(total_steps)

    log(f"Output Dir: {OUTPUT_DIR}")
    log(f"Refresh Perf: {refresh_perf_result}")
    log(f"N Runs: {n_runs}")

    for test_case in test_cases:
        bench_test_case(test_case)

    # 结束进度条
    pb_finish()

    return 0

if __name__ == "__main__":
    sys.exit(main())
