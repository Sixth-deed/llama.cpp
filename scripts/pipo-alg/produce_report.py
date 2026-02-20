#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
局部最优参数区域分析 - 找出 decode time 显著较小的连续参数空间
"""

import re
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
import os

@dataclass
class Result:
    alpha: float
    beta: float
    theta: float
    decode_time: float
    prefill_time: float
    total_time: float
    total_cuda_mem: float

def parse_data(filepath: str) -> List[Result]:
    """解析 Python 对象格式的输出"""
    results = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取 results 列表部分
    match = re.search(r'results:\s*\[(.*?)\]', content, re.DOTALL)
    if not match:
        return results
    
    # 解析每个 Result 对象
    result_pattern = r'Result\(total_time=([\d.]+),\s*total_cuda_mem=([\d.]+),\s*decode_time=([\d.]+),\s*prefill_time=([\d.]+),\s*alg_config=AlgConfig\(alpha=([\d.]+),\s*beta=([\d.]+),\s*theta=([\d.]+)\)\)'
    
    matches = re.findall(result_pattern, content)
    
    for match in matches:
        results.append(Result(
            alpha=float(match[4]),
            beta=float(match[5]),
            theta=float(match[6]),
            decode_time=float(match[2]),
            prefill_time=float(match[3]),
            total_time=float(match[0]),
            total_cuda_mem=float(match[1])
        ))
    
    return results

def find_local_optimal_region(results: List[Result], 
                               percentile: float = 20,
                               min_neighbors: int = 2) -> Dict:
    """
    找出局部最优参数区域
    
    Args:
        results: 所有测试结果
        percentile: 取前百分之多少作为"较小"的阈值
        min_neighbors: 至少需要多少个相邻点都在最优区域内
    
    Returns:
        包含最优区域信息的字典
    """
    # 计算 decode time 的阈值（前 percentile%）
    decode_times = [r.decode_time for r in results]
    threshold = np.percentile(decode_times, percentile)
    
    # 找出所有低于阈值的配置
    optimal_configs = [r for r in results if r.decode_time <= threshold]
    
    # 按参数分组，找出连续区域
    alpha_values = sorted(set(r.alpha for r in results))
    beta_values = sorted(set(r.beta for r in results))
    theta_values = sorted(set(r.theta for r in results))
    
    # 创建参数空间网格
    grid = {}
    for r in results:
        key = (r.alpha, r.beta, r.theta)
        grid[key] = r
    
    # 找出局部最优区域（相邻点也在最优区域内）
    def get_neighbors(alpha, beta, theta):
        """获取相邻的参数配置"""
        neighbors = []
        alpha_idx = alpha_values.index(alpha)
        beta_idx = beta_values.index(beta)
        theta_idx = theta_values.index(theta)
        
        # 检查 6 个方向的邻居
        directions = [
            (alpha_idx - 1, beta_idx, theta_idx),
            (alpha_idx + 1, beta_idx, theta_idx),
            (alpha_idx, beta_idx - 1, theta_idx),
            (alpha_idx, beta_idx + 1, theta_idx),
            (alpha_idx, beta_idx, theta_idx - 1),
            (alpha_idx, beta_idx, theta_idx + 1),
        ]
        
        for a_idx, b_idx, t_idx in directions:
            if 0 <= a_idx < len(alpha_values) and \
               0 <= b_idx < len(beta_values) and \
               0 <= t_idx < len(theta_values):
                neighbor_key = (alpha_values[a_idx], beta_values[b_idx], theta_values[t_idx])
                if neighbor_key in grid:
                    neighbors.append(grid[neighbor_key])
        
        return neighbors
    
    # 识别局部最优区域
    local_optimal_regions = []
    optimal_set = set((r.alpha, r.beta, r.theta) for r in optimal_configs)
    
    for r in optimal_configs:
        neighbors = get_neighbors(r.alpha, r.beta, r.theta)
        optimal_neighbors = [n for n in neighbors if (n.alpha, n.beta, n.theta) in optimal_set]
        
        if len(optimal_neighbors) >= min_neighbors:
            local_optimal_regions.append({
                'config': r,
                'optimal_neighbor_count': len(optimal_neighbors),
                'neighbors': optimal_neighbors
            })
    
    # 统计区域信息
    region_stats = {
        'threshold': threshold,
        'total_optimal_configs': len(optimal_configs),
        'local_optimal_regions': len(local_optimal_regions),
        'optimal_configs': optimal_configs,
        'local_optimal_points': local_optimal_regions,
        'alpha_range': (min(r.alpha for r in optimal_configs), 
                       max(r.alpha for r in optimal_configs)),
        'beta_range': (min(r.beta for r in optimal_configs), 
                      max(r.beta for r in optimal_configs)),
        'theta_range': (min(r.theta for r in optimal_configs), 
                       max(r.theta for r in optimal_configs)),
    }
    
    return region_stats

def analyze_parameter_sensitivity(results: List[Result]) -> Dict:
    """分析各参数对 decode time 的敏感度"""
    # 按参数分组
    alpha_groups = defaultdict(list)
    beta_groups = defaultdict(list)
    theta_groups = defaultdict(list)
    
    for r in results:
        alpha_groups[r.alpha].append(r.decode_time)
        beta_groups[r.beta].append(r.decode_time)
        theta_groups[r.theta].append(r.decode_time)
    
    # 计算每个参数值的统计信息
    sensitivity = {}
    for param_name, groups in [('alpha', alpha_groups), 
                                ('beta', beta_groups), 
                                ('theta', theta_groups)]:
        stats_data = {}
        for value, times in sorted(groups.items()):
            stats_data[value] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'range': np.max(times) - np.min(times),
                'count': len(times)
            }
        sensitivity[param_name] = stats_data
    
    return sensitivity

def plot_optimal_region(results: List[Result], 
                        region_stats: Dict,
                        save_path: str = 'optimal_region.png'):
    """绘制最优参数区域"""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('局部最优参数区域分析', fontsize=16, fontweight='bold')
    
    # 1. 3D 散点图 - 参数空间分布
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # 所有配置
    alphas = [r.alpha for r in results]
    betas = [r.beta for r in results]
    thetas = [r.theta for r in results]
    decode_times = [r.decode_time for r in results]
    
    # 归一化颜色
    norm = plt.Normalize(min(decode_times), max(decode_times))
    cmap = plt.cm.viridis
    
    scatter = ax1.scatter(alphas, betas, thetas, 
                         c=decode_times, cmap=cmap, norm=norm,
                         s=50, alpha=0.6, edgecolors='black')
    
    # 标记最优区域
    optimal_alphas = [r.alpha for r in region_stats['optimal_configs']]
    optimal_betas = [r.beta for r in region_stats['optimal_configs']]
    optimal_thetas = [r.theta for r in region_stats['optimal_configs']]
    
    ax1.scatter(optimal_alphas, optimal_betas, optimal_thetas,
               c='red', s=100, marker='*', label='最优区域', edgecolors='black')
    
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Beta')
    ax1.set_zlabel('Theta')
    ax1.set_title('参数空间分布\n(颜色=Decode Time)')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Decode Time (ms)')
    
    # 2. Alpha-Beta 平面投影
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(alphas, betas, c=decode_times, 
                          cmap=cmap, norm=norm, s=80, alpha=0.7, edgecolors='black')
    ax2.scatter(optimal_alphas, optimal_betas, c='red', s=150, 
               marker='*', edgecolors='black', label='最优区域')
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('Beta')
    ax2.set_title('Alpha-Beta 平面投影')
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='Decode Time (ms)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Alpha-Theta 平面投影
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(alphas, thetas, c=decode_times, 
                          cmap=cmap, norm=norm, s=80, alpha=0.7, edgecolors='black')
    ax3.scatter(optimal_alphas, optimal_thetas, c='red', s=150, 
               marker='*', edgecolors='black', label='最优区域')
    ax3.set_xlabel('Alpha')
    ax3.set_ylabel('Theta')
    ax3.set_title('Alpha-Theta 平面投影')
    ax3.legend()
    plt.colorbar(scatter3, ax=ax3, label='Decode Time (ms)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Beta-Theta 平面投影
    ax4 = fig.add_subplot(2, 3, 4)
    scatter4 = ax4.scatter(betas, thetas, c=decode_times, 
                          cmap=cmap, norm=norm, s=80, alpha=0.7, edgecolors='black')
    ax4.scatter(optimal_betas, optimal_thetas, c='red', s=150, 
               marker='*', edgecolors='black', label='最优区域')
    ax4.set_xlabel('Beta')
    ax4.set_ylabel('Theta')
    ax4.set_title('Beta-Theta 平面投影')
    ax4.legend()
    plt.colorbar(scatter4, ax=ax4, label='Decode Time (ms)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Decode Time 分布直方图
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(decode_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(region_stats['threshold'], color='red', linestyle='--', 
               linewidth=2, label=f'阈值 ({region_stats["threshold"]:.2f} ms)')
    
    # 标记最优区域的分布
    optimal_times = [r.decode_time for r in region_stats['optimal_configs']]
    ax5.hist(optimal_times, bins=20, alpha=0.7, color='red', edgecolor='black',
            label=f'最优区域 (n={len(optimal_times)})')
    
    ax5.set_xlabel('Decode Time (ms)')
    ax5.set_ylabel('频数')
    ax5.set_title('Decode Time 分布')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 参数敏感度分析
    ax6 = fig.add_subplot(2, 3, 6)
    sensitivity = analyze_parameter_sensitivity(results)
    
    params = ['alpha', 'beta', 'theta']
    avg_ranges = []
    for param in params:
        ranges = [stats_data['range'] for stats_data in sensitivity[param].values()]
        avg_ranges.append(np.mean(ranges))
    
    colors = ['blue', 'green', 'orange']
    bars = ax6.bar(params, avg_ranges, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('平均 Decode Time 变化范围 (ms)')
    ax6.set_title('参数敏感度分析\n(范围越大越敏感)')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上标注数值
    for bar, val in zip(bars, avg_ranges):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"最优区域图已保存：{save_path}")
    plt.close()

def generate_region_report(results: List[Result], 
                           region_stats: Dict,
                           output_path: str = 'optimal_region_report.md'):
    """生成最优区域分析报告"""
    
    report = []
    report.append("# 局部最优参数区域分析报告\n")
    
    report.append("## 1. 最优区域定义\n")
    report.append(f"- **阈值标准**: Decode Time ≤ {region_stats['threshold']:.2f} ms (前 20%)")
    report.append(f"- **最优配置数量**: {region_stats['total_optimal_configs']} / {len(results)}")
    report.append(f"- **最优配置占比**: {region_stats['total_optimal_configs']/len(results)*100:.1f}%\n")
    
    report.append("## 2. 最优参数空间范围\n")
    report.append(f"| 参数 | 最小值 | 最大值 | 范围 |")
    report.append(f"|------|--------|--------|------|")
    report.append(f"| Alpha | {region_stats['alpha_range'][0]} | {region_stats['alpha_range'][1]} | {region_stats['alpha_range'][1] - region_stats['alpha_range'][0]} |")
    report.append(f"| Beta | {region_stats['beta_range'][0]} | {region_stats['beta_range'][1]} | {region_stats['beta_range'][1] - region_stats['beta_range'][0]} |")
    report.append(f"| Theta | {region_stats['theta_range'][0]} | {region_stats['theta_range'][1]} | {region_stats['theta_range'][1] - region_stats['theta_range'][0]} |")
    report.append("")
    
    report.append("## 3. 最优配置详情\n")
    report.append("### 3.1 Top 10 最优配置\n")
    sorted_optimal = sorted(region_stats['optimal_configs'], key=lambda x: x.decode_time)
    report.append("| Rank | Alpha | Beta | Theta | Decode Time | Total Time | CUDA Mem |")
    report.append("|------|-------|------|-------|-------------|------------|----------|")
    for i, r in enumerate(sorted_optimal[:10], 1):
        report.append(f"| {i} | {r.alpha} | {r.beta} | {r.theta} | {r.decode_time:.2f} | {r.total_time:.2f} | {r.total_cuda_mem:.2f} |")
    report.append("")
    
    report.append("### 3.2 局部最优点（相邻点也在最优区域内）\n")
    if region_stats['local_optimal_points']:
        report.append(f"- **局部最优点数量**: {len(region_stats['local_optimal_points'])}\n")
        report.append("| Alpha | Beta | Theta | Decode Time | 最优邻居数 |")
        report.append("|-------|------|-------|-------------|------------|")
        for point in sorted(region_stats['local_optimal_points'], 
                          key=lambda x: x['config'].decode_time)[:10]:
            r = point['config']
            report.append(f"| {r.alpha} | {r.beta} | {r.theta} | {r.decode_time:.2f} | {point['optimal_neighbor_count']} |")
    else:
        report.append("未找到满足条件的局部最优点\n")
    report.append("")
    
    report.append("## 4. 参数敏感度分析\n")
    sensitivity = analyze_parameter_sensitivity(results)
    
    for param in ['alpha', 'beta', 'theta']:
        report.append(f"### 4.1 {param.upper()} 敏感度")
        report.append(f"| 值 | 平均 Decode Time | 标准差 | 最小值 | 最大值 | 变化范围 |")
        report.append(f"|-----|-----------------|--------|--------|--------|----------|")
        for value, stats_data in sorted(sensitivity[param].items()):
            report.append(f"| {value} | {stats_data['mean']:.2f} | {stats_data['std']:.2f} | {stats_data['min']:.2f} | {stats_data['max']:.2f} | {stats_data['range']:.2f} |")
        report.append("")
    
    report.append("## 5. 推荐参数区域\n")
    report.append("基于分析，推荐的参数空间区域为：\n")
    report.append(f"```\n")
    report.append(f"Alpha:  {region_stats['alpha_range'][0]} ~ {region_stats['alpha_range'][1]}\n")
    report.append(f"Beta:   {region_stats['beta_range'][0]} ~ {region_stats['beta_range'][1]}\n")
    report.append(f"Theta:  {region_stats['theta_range'][0]} ~ {region_stats['theta_range'][1]}\n")
    report.append(f"```\n")
    
    # 找出最稳定的区域（decode time 变化最小）
    report.append("### 5.1 最稳定子区域\n")
    
    # 按 beta 分组分析
    beta_groups = defaultdict(list)
    for r in region_stats['optimal_configs']:
        beta_groups[r.beta].append(r)
    
    most_stable_beta = min(beta_groups.keys(), 
                          key=lambda b: np.std([r.decode_time for r in beta_groups[b]]))
    
    report.append(f"- **最稳定的 Beta 值**: {most_stable_beta}")
    report.append(f"- **该 Beta 下的配置数**: {len(beta_groups[most_stable_beta])}")
    report.append(f"- **平均 Decode Time**: {np.mean([r.decode_time for r in beta_groups[most_stable_beta]]):.2f} ms")
    report.append(f"- **Decode Time 标准差**: {np.std([r.decode_time for r in beta_groups[most_stable_beta]]):.2f} ms\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"分析报告已生成：{output_path}")
    return '\n'.join(report)

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SCRIPT_DIR)
    filepath = 'grd_result1.log'
    
    print("=" * 70)
    print("局部最优参数区域分析")
    print("=" * 70)
    
    # 解析数据
    results = parse_data(filepath)
    print(f"\n成功解析 {len(results)} 条记录\n")
    
    # 找出局部最优区域
    print("分析局部最优区域...")
    region_stats = find_local_optimal_region(results, percentile=20, min_neighbors=2)
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("最优区域摘要")
    print("=" * 70)
    print(f"阈值 (前 20%): {region_stats['threshold']:.2f} ms")
    print(f"最优配置数：{region_stats['total_optimal_configs']} / {len(results)}")
    print(f"\n最优参数空间范围:")
    print(f"  Alpha: {region_stats['alpha_range'][0]} ~ {region_stats['alpha_range'][1]}")
    print(f"  Beta:  {region_stats['beta_range'][0]} ~ {region_stats['beta_range'][1]}")
    print(f"  Theta: {region_stats['theta_range'][0]} ~ {region_stats['theta_range'][1]}")
    
    # 显示 Top 5 最优配置
    print("\nTop 5 最优配置:")
    sorted_optimal = sorted(region_stats['optimal_configs'], key=lambda x: x.decode_time)
    for i, r in enumerate(sorted_optimal[:5], 1):
        print(f"  {i}. alpha={r.alpha}, beta={r.beta}, theta={r.theta} -> {r.decode_time:.2f} ms")
    
    # 生成报告
    print("\n生成详细报告...")
    generate_region_report(results, region_stats)
    
    # 绘制图表
    print("生成可视化图表...")
    plot_optimal_region(results, region_stats)
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()