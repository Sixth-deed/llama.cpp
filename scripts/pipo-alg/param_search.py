#!/usr/bin/env python3
from dataclasses import dataclass
import os
import sys
import subprocess
import re
import tempfile
import time
import json  # 新增导入
from typing import Optional, List, Dict
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
    alpha : float = 1.0
    beta : float = 1.0
    theta : float = 0.1

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
        
    def update(self, steps: int = 1):
        """Update progress by specific number of steps"""
        self.current += steps
        if self.current > self.total:
            self.current = self.total
            
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

def single_run(alg_config : AlgConfig, n_runs: int, progress_bar: Optional[ProgressBar] = None, cache: Optional[Dict[str, Result]] = None) -> Optional[Result]:
    """
        Args:
        alg_config: the configuration of the algorithm
        n_runs: number of runs
        progress_bar: optional progress bar to update per run
        cache: dictionary to cache results based on algorithm output hash

        Returns:
            A Result object containing the results of the run, or None if failed.
    """
    # 1. Run pipo-alg to generate graph config
    alg_cmd = [
        ALG_BIN, 
        MODEL_PATH, 
        "-alpha", str(alg_config.alpha), 
        "-beta", str(alg_config.beta),
        "-theta", str(alg_config.theta),
        # "-greedy"
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
    
    # 2. Generate Cache Key (Normalized JSON)
    # We normalize JSON to ensure whitespace differences don't cause cache misses
    cache_key = None
    if cache is not None:
        try:
            json_obj = json.loads(graph_json)
            cache_key = json.dumps(json_obj, sort_keys=True)
        except json.JSONDecodeError:
            # If not valid JSON, use raw string hash
            cache_key = graph_json
            
        # 3. Check Cache
        if cache_key in cache:
            # Cache Hit: Reuse previous result
            cached_result = cache[cache_key]
            # Create a new Result object with current config but cached metrics
            res = Result(
                total_time=cached_result.total_time,
                total_cuda_mem=cached_result.total_cuda_mem,
                decode_time=cached_result.decode_time,
                prefill_time=cached_result.prefill_time,
                alg_config=alg_config  # Update to current config for reporting
            )
            # Update progress bar for the skipped runs
            if progress_bar is not None:
                progress_bar.update(n_runs)
            return res

    # 4. Create temporary file for graph config (Only if cache miss)
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_file.write(graph_json)
        temp_file.close() # Close so llama-simple can open it
        
        # 5. Run llama-simple n_runs times
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
                progress_bar.update(1)
            
        # 6. Calculate averages
        avg_total_time = sum(m['total_time'] for m in collected_metrics) / len(collected_metrics)
        avg_total_cuda_mem = sum(m['total_cuda_mem'] for m in collected_metrics) / len(collected_metrics)
        avg_decode_time = sum(m['decode_time'] for m in collected_metrics) / len(collected_metrics)
        avg_prefill_time = sum(m['prefill_time'] for m in collected_metrics) / len(collected_metrics)
        
        result = Result(
            total_time=avg_total_time,
            total_cuda_mem=avg_total_cuda_mem,
            decode_time=avg_decode_time,
            prefill_time=avg_prefill_time,
            alg_config=alg_config
        )
        
        # 7. Store in Cache
        if cache is not None and cache_key is not None:
            cache[cache_key] = result
            
        return result
        
    finally:
        # 8. Cleanup temp file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def main():
    alpha_list = [1.0]
    beta_list = [0.8, 1.0, 1.2, 1.3, 1.4]
    theta_list = [0.5]
    n_runs = 5
    
    results = []
    # Initialize Cache
    result_cache = {}
    
    # Calculate total runs for progress bar (per run level, not per config)
    total_iterations = len(alpha_list) * len(theta_list) * len(beta_list) * n_runs
    progress = ProgressBar(total_iterations)
    
    for alpha in alpha_list:
        for beta in beta_list: 
            for theta in theta_list:
                config = AlgConfig()
                config.alpha = alpha
                config.theta = theta
                config.beta = beta
                
                res = single_run(config, n_runs, progress_bar=progress, cache=result_cache)
                if res is not None:
                    results.append(res)
                else:
                    # Skip failed configs - progress already updated in single_run
                    pass
            
    progress.finish()
    
    print("results:\n", results)

    # Print results to stdout
    for r in results:
        print(f"[alpha={r.alg_config.alpha}, beta={r.alg_config.beta}, theta = {r.alg_config.theta}]")
        print("{")
        print(f"\tDecode time per token = {r.decode_time:.2f} ms")
        print(f"\tPrefill time = {r.prefill_time:.2f} ms")
        print(f"\tTotal time = {r.total_time:.2f} ms")
        print(f"\tTotal CUDA Mem = {r.total_cuda_mem:.2f} MiB")
        print("}")
        print("")
    

if __name__ == "__main__":
    main()