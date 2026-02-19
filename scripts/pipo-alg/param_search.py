#!/usr/bin/env python3
from dataclasses import dataclass
import os
import sys
import subprocess
import re
import tempfile
import time
from typing import Optional, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir + "/../../")

ALG_BIN="build-release/bin/pipo-alg"
TARGET_BIN="build-release/bin/llama-simple"
MODEL_PATH="../model/Qwen3-14B-Q4_K_M.gguf"

@dataclass
class AlgConfig:
    alpha = 1.0
    theta = 0.1

@dataclass
class AlgOutput:
    json_str : str

@dataclass
class Result:
    # milliseconds
    total_time: float
    # Mbytes
    total_cuda_mem: float
    # decode time per token (milliseconds)
    decode_time: float
    # prefill time (milliseconds)
    prefill_time: float
    
    alg_config: AlgConfig

class ProgressBar:
    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.start_time = time.time()
        
    def update(self):
        self.current += 1
        elapsed = time.time() - self.start_time
        if self.current == 0:
            return
        avg_time = elapsed / self.current
        remaining = (self.total - self.current) * avg_time
        
        percent = (self.current / self.total) * 100
        bar_length = 40
        filled_length = int(bar_length * self.current // self.total)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        
        # Format ETA as minutes and seconds
        remaining_mins = int(remaining // 60)
        remaining_secs = int(remaining % 60)
        eta_str = f"{remaining_mins}m {remaining_secs}s"
        
        # Print to stderr to avoid interfering with stdout data
        sys.stderr.write(f'\r[{bar}] {percent:.1f}% | Run: {self.current}/{self.total} | ETA: {eta_str}')
        sys.stderr.flush()
        
    def finish(self):
        sys.stderr.write('\n')
        sys.stderr.flush()

def parse_metrics(stderr_str: str) -> Optional[dict]:
    """
    Parse metrics from llama-simple stderr output.
    Returns a dict with keys: total_time, total_cuda_mem, decode_time, prefill_time
    Returns None if parsing fails.
    """
    metrics = {}
    
    # Parse CUDA0 model buffer size
    model_mem_match = re.search(r"CUDA0 model buffer size =\s*([\d.]+)\s*MiB", stderr_str)
    if not model_mem_match:
        return None
    metrics['model_mem'] = float(model_mem_match.group(1))
    
    # Parse CUDA0 KV buffer size
    kv_mem_match = re.search(r"CUDA0 KV buffer size =\s*([\d.]+)\s*MiB", stderr_str)
    if not kv_mem_match:
        return None
    metrics['kv_mem'] = float(kv_mem_match.group(1))
    
    # Parse CUDA0 compute buffer size
    compute_mem_match = re.search(r"CUDA0 compute buffer size =\s*([\d.]+)\s*MiB", stderr_str)
    if not compute_mem_match:
        return None
    metrics['compute_mem'] = float(compute_mem_match.group(1))
    
    # Calculate total cuda mem
    metrics['total_cuda_mem'] = metrics['model_mem'] + metrics['kv_mem'] + metrics['compute_mem']
    
    # Parse prompt eval time (prefill time)
    # Example: prompt eval time =     615.74 ms /    99 tokens
    prefill_match = re.search(r"prompt eval time =\s*([\d.]+)\s*ms", stderr_str)
    if not prefill_match:
        return None
    metrics['prefill_time'] = float(prefill_match.group(1))
    
    # Parse eval time per token (decode time)
    # Example: eval time =    7486.40 ms /    31 runs   (  241.50 ms per token,     4.14 tokens per second)
    decode_match = re.search(r"(?<!prompt )eval time\s*=.*?\(\s*([\d.]+)\s*ms per token", stderr_str)
    if not decode_match:
        return None
    metrics['decode_time'] = float(decode_match.group(1))
    
    # Parse total time
    # Example: total time =   21202.20 ms /   130 tokens
    total_time_match = re.search(r"total time =\s*([\d.]+)\s*ms", stderr_str)
    if not total_time_match:
        return None
    metrics['total_time'] = float(total_time_match.group(1))
    
    return metrics

def single_run(alg_config : AlgConfig, n_runs: int, progress_bar: Optional[ProgressBar] = None) -> Optional[Result]:
    """
        Args:
        alg_config: the configuration of the algorithm
        n_runs: number of runs
        progress_bar: optional progress bar to update per run

        Returns:
            A Result object containing the results of the run, or None if failed.
    """
    # 1. Run pipo-alg to generate graph config
    alg_cmd = [
        ALG_BIN, 
        MODEL_PATH, 
        "-alpha", str(alg_config.alpha), 
        "-theta", str(alg_config.theta)
    ]
    
    try:
        pipo_proc = subprocess.run(alg_cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        print(f"Error running pipo-alg: {e}", file=sys.stderr)
        return None
        
    if pipo_proc.returncode != 0:
        print(pipo_proc.stderr, file=sys.stderr)
        return None
        
    graph_json = pipo_proc.stdout
    
    # 2. Create temporary file for graph config
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_file.write(graph_json)
        temp_file.close() # Close so llama-simple can open it
        
        # 3. Run llama-simple n_runs times
        collected_metrics = []
        
        for i in range(n_runs):
            target_cmd = [
                TARGET_BIN,
                "-m", MODEL_PATH,
                "-pipo", temp_file.name
            ]
            
            try:
                llama_proc = subprocess.run(target_cmd, capture_output=True, text=True, check=False)
            except Exception as e:
                print(f"Error running llama-simple (run {i+1}): {e}", file=sys.stderr)
                return None
                
            if llama_proc.returncode != 0:
                print(llama_proc.stderr, file=sys.stderr)
                return None
                
            metrics = parse_metrics(llama_proc.stderr)
            if metrics is None:
                print(f"Failed to parse metrics from run {i+1}", file=sys.stderr)
                print(llama_proc.stderr, file=sys.stderr)
                return None
                
            collected_metrics.append(metrics)
            
            # Update progress bar per run
            if progress_bar is not None:
                progress_bar.update()
            
        # 4. Calculate averages
        avg_total_time = sum(m['total_time'] for m in collected_metrics) / len(collected_metrics)
        avg_total_cuda_mem = sum(m['total_cuda_mem'] for m in collected_metrics) / len(collected_metrics)
        avg_decode_time = sum(m['decode_time'] for m in collected_metrics) / len(collected_metrics)
        avg_prefill_time = sum(m['prefill_time'] for m in collected_metrics) / len(collected_metrics)
        
        return Result(
            total_time=avg_total_time,
            total_cuda_mem=avg_total_cuda_mem,
            decode_time=avg_decode_time,
            prefill_time=avg_prefill_time,
            alg_config=alg_config
        )
        
    finally:
        # 5. Cleanup temp file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def main():
    alpha_list = [0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8]
    theta_list = [0.4, 0.5, 0.6, 0.8, 1]
    n_runs = 5
    
    results = []
    
    # Calculate total runs for progress bar (per run level, not per config)
    total_iterations = len(alpha_list) * len(theta_list) * n_runs
    progress = ProgressBar(total_iterations)
    
    for alpha in alpha_list:
        for theta in theta_list:
            config = AlgConfig()
            config.alpha = alpha
            config.theta = theta
            res = single_run(config, n_runs, progress_bar=progress)
            if res is not None:
                results.append(res)
            else:
                # Skip failed configs - progress already updated in single_run
                pass
            
    progress.finish()
    
    print("results:\n", results)

    # Print results to stdout
    for r in results:
        print(f"[alpha={r.alg_config.alpha}, theta = {r.alg_config.theta}]")
        print("{")
        print(f"\tDecode time per token = {r.decode_time:.2f} ms")
        print(f"\tPrefill time = {r.prefill_time:.2f} ms")
        print(f"\tTotal time = {r.total_time:.2f} ms")
        print(f"\tTotal CUDA Mem = {r.total_cuda_mem:.2f} MiB")
        print("}")
        print("")
    if results:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取排序后的唯一参数值
        unique_alphas = sorted(list(set(alpha_list)))
        unique_thetas = sorted(list(set(theta_list)))
        
        # 创建参数值到等间距整数索引的映射
        alpha_to_idx = {alpha: i for i, alpha in enumerate(unique_alphas)}
        theta_to_idx = {theta: i for i, theta in enumerate(unique_thetas)}
        
        # 将原始参数值转换为等间距的整数坐标
        x_positions = np.array([alpha_to_idx[r.alg_config.alpha] for r in results])
        y_positions = np.array([theta_to_idx[r.alg_config.theta] for r in results])
        decode_times = np.array([r.decode_time for r in results])
        
        # 设置柱子尺寸：横截面 0.7 x 0.7，相邻间距为1
        bar_width = 0.7   # 柱子宽度
        bar_depth = 0.7   # 柱子深度
        
        # 计算每个柱子的左下角坐标
        # 柱子中心在整数位置，左下角需要偏移宽度/2
        x_left = x_positions - bar_width / 2
        y_left = y_positions - bar_depth / 2
        
        # 颜色映射
        colors = plt.cm.viridis(decode_times / np.max(decode_times))
        
        # 绘制3D柱状图
        ax.bar3d(x_left, y_left, np.zeros_like(decode_times), 
                 bar_width, bar_depth, decode_times, 
                 color=colors, shade=True)
        
        # 设置坐标轴标签
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Theta')
        ax.set_zlabel('Decode Time (ms)')
        ax.set_title('Decode Time vs Alpha & Theta (3D Bar)')
        
        # 设置刻度：使用整数位置，标签显示原始参数值
        ax.set_xticks(range(len(unique_alphas)))
        ax.set_xticklabels(unique_alphas)
        ax.set_yticks(range(len(unique_thetas)))
        ax.set_yticklabels(unique_thetas)
        
        # 设置视角以获得更好的视觉效果
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show() 
    

if __name__ == "__main__":
    main()