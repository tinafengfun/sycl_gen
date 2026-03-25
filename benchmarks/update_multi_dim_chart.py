#!/usr/bin/env python3
"""
Updated Multi-Dimensional Analysis for All 23 Kernels
4-panel layout matching the original chart format
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style matching original charts
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold'

# All 23 kernels organized by category
KERNELS = {
    'Winograd': [
        'winograd_output_relu_input',
        'winograd_filter_transform', 
        'winograd_real',
        'winograd_input_transform'
    ],
    'Element-wise': [
        'global_scale',
        'add_bias_batched',
        'add_bias_nchw',
        'add_vectors',
        'copy_type_converted',
        'global_scale_fp16_nhwc',
        'expand_planes_nhwc',
        'expand_planes_nchw',
        'expand_planes_fp16_nhwc',
        'add_vectors_hnc_nhc'
    ],
    'Normalization': [
        'hard_batch_norm',
        'layer_norm'
    ],
    'Reduction': [
        'global_avg_pool_real',
        'global_avg_pool_nhwc_fp16',
        'softmax_real'
    ],
    'Complex Fused': [
        'se_layer_nhwc',
        'hard_fused_kernel'
    ],
    'Layout/Gather': [
        'nchw_to_nhwc',
        'policy_map'
    ]
}

# Color mapping matching original scheme
CATEGORY_COLORS = {
    'Winograd': '#e74c3c',      # Red
    'Element-wise': '#3498db',   # Blue
    'Normalization': '#2ecc71',  # Green
    'Reduction': '#f39c12',      # Orange
    'Complex Fused': '#9b59b6',  # Purple
    'Layout/Gather': '#1abc9c'   # Teal
}

def parse_csv(filepath):
    """Parse a kernel result CSV file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        header_idx = 0
        for i, line in enumerate(lines):
            if 'Version' in line:
                header_idx = i
                break
        
        data = []
        for line in lines[header_idx+1:]:
            line = line.strip()
            if not line or line.startswith('Device'):
                continue
            
            parts = line.split(',')
            if len(parts) >= 3:
                row = {'Version': parts[0]}
                for part in parts[1:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        try:
                            row[key] = float(value)
                        except:
                            row[key] = value
                data.append(row)
        
        return pd.DataFrame(data) if data else None
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def extract_metrics(df, kernel_name, category):
    """Extract key metrics from kernel data."""
    if df is None or df.empty:
        return None
    
    result = {
        'kernel': kernel_name,
        'category': category,
        'v0_gflops': 0,
        'best_gflops': 0,
        'speedup': 1.0,
        'best_version': 'V0'
    }
    
    # Extract GFLOPS data
    if 'GFLOPS' in df.columns:
        df['GFLOPS'] = pd.to_numeric(df['GFLOPS'], errors='coerce')
        
        # Get V0 baseline
        v0_data = df[df['Version'] == 'V0']
        if not v0_data.empty:
            result['v0_gflops'] = v0_data['GFLOPS'].max()
        
        # Get best performance
        best_idx = df['GFLOPS'].idxmax()
        result['best_gflops'] = df.loc[best_idx, 'GFLOPS']
        result['best_version'] = df.loc[best_idx, 'Version']
        
        if result['v0_gflops'] > 0:
            result['speedup'] = result['best_gflops'] / result['v0_gflops']
    
    # Extract bandwidth data
    if 'Bandwidth_GB/s' in df.columns:
        df['Bandwidth_GB/s'] = pd.to_numeric(df['Bandwidth_GB/s'], errors='coerce')
        result['bandwidth_util'] = df['Bandwidth_GB/s'].max() / 700 * 100  # % of theoretical
    
    return result

def determine_technique(kernel_name, speedup, best_version):
    """Determine which optimization technique was most effective."""
    if speedup > 5.0:
        return 'Single-thread Mode'
    elif speedup > 2.0:
        return 'Loop Unrolling'
    elif speedup > 1.2:
        return 'WG Size Tuning'
    elif speedup > 1.05:
        return 'Vectorization'
    else:
        return 'Baseline (optimal)'

def create_multi_dimensional_analysis():
    """Create the 4-panel multi-dimensional analysis chart."""
    results_dir = Path('/home/intel/tianfeng/opencode_bench/benchmarks/results')
    
    # Collect all data
    all_data = []
    for category, kernels in KERNELS.items():
        for kernel in kernels:
            filepath = results_dir / f"{kernel}_results.csv"
            df = parse_csv(filepath)
            metrics = extract_metrics(df, kernel, category)
            if metrics:
                metrics['technique'] = determine_technique(
                    kernel, metrics['speedup'], metrics['best_version']
                )
                all_data.append(metrics)
    
    df_all = pd.DataFrame(all_data)
    
    # Create 4-panel figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Comprehensive Performance Analysis - All 23 Kernels\nIntel Xe2 (Battlemage G21)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Panel 1: Optimization Speedup Comparison
    speedup_data = df_all.sort_values('speedup', ascending=True)
    colors1 = [CATEGORY_COLORS[cat] for cat in speedup_data['category']]
    
    bars1 = ax1.barh(range(len(speedup_data)), speedup_data['speedup'], color=colors1, alpha=0.8)
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax1.set_xlabel('Speedup Ratio (Best / Baseline)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Kernel', fontsize=11, fontweight='bold')
    ax1.set_title('Optimization Speedup Comparison\nHigher is Better', fontsize=13, fontweight='bold')
    ax1.set_yticks(range(len(speedup_data)))
    ax1.set_yticklabels([k.replace('_', '_\n') if len(k) > 15 else k for k in speedup_data['kernel']], fontsize=8)
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels for top 5
    for i, (idx, row) in enumerate(speedup_data.tail(5).iterrows()):
        actual_idx = len(speedup_data) - 5 + i
        ax1.text(row['speedup'] + 0.1, actual_idx, f"{row['speedup']:.2f}x", 
                va='center', fontsize=8, fontweight='bold')
    
    # Panel 2: Absolute Performance Comparison
    perf_data = df_all.sort_values('best_gflops', ascending=False).head(16)  # Top 16 for visibility
    x = np.arange(len(perf_data))
    width = 0.35
    
    bars_v0 = ax2.bar(x - width/2, perf_data['v0_gflops'], width, label='V0 Baseline', 
                     color='#3498db', alpha=0.8, edgecolor='black')
    bars_best = ax2.bar(x + width/2, perf_data['best_gflops'], width, label='Best Optimized', 
                       color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Kernel', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Performance (GFLOPS)', fontsize=11, fontweight='bold')
    ax2.set_title('Absolute Performance Comparison\nTop 16 Kernels', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([k.replace('_', '_\n') for k in perf_data['kernel']], 
                       rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Panel 3: Distribution of Best Optimization Techniques
    technique_counts = df_all['technique'].value_counts()
    colors3 = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c'][:len(technique_counts)]
    
    wedges, texts, autotexts = ax3.pie(technique_counts.values, labels=technique_counts.index, 
                                        autopct='%1.0f%%', colors=colors3, startangle=90,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title('Distribution of Best Optimization\nTechniques (23 Kernels)', fontsize=13, fontweight='bold')
    
    # Panel 4: Memory Bandwidth Utilization
    # Check if bandwidth_util column exists and has valid data
    if 'bandwidth_util' not in df_all.columns:
        df_all['bandwidth_util'] = 0
    
    bandwidth_data = df_all[df_all['bandwidth_util'] > 0].sort_values('bandwidth_util', ascending=True)
    colors4 = [CATEGORY_COLORS[cat] for cat in bandwidth_data['category']]
    
    bars4 = ax4.barh(range(len(bandwidth_data)), bandwidth_data['bandwidth_util'], color=colors4, alpha=0.8)
    ax4.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='50% target')
    ax4.axvline(x=80, color='green', linestyle='--', linewidth=2, label='80% excellent')
    
    ax4.set_xlabel('Bandwidth Utilization (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Kernel', fontsize=11, fontweight='bold')
    ax4.set_title('Memory Bandwidth Utilization\n(Higher is Better)', fontsize=13, fontweight='bold')
    ax4.set_yticks(range(len(bandwidth_data)))
    ax4.set_yticklabels(bandwidth_data['kernel'], fontsize=8)
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (idx, row) in enumerate(bandwidth_data.iterrows()):
        ax4.text(row['bandwidth_util'] + 1, i, f"{row['bandwidth_util']:.1f}%", 
                va='center', fontsize=8)
    
    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CATEGORY_COLORS[cat], label=cat) for cat in CATEGORY_COLORS]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=10, 
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('/home/intel/tianfeng/opencode_bench/benchmarks/charts/multi_dimensional_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Updated: multi_dimensional_analysis.png")
    plt.close()
    
    return df_all

def main():
    """Generate updated multi-dimensional analysis."""
    print("="*60)
    print("Updating Multi-Dimensional Analysis")
    print("All 23 Kernels")
    print("="*60)
    
    df = create_multi_dimensional_analysis()
    
    print("\nSummary Statistics:")
    print(f"- Total kernels: {len(df)}")
    print(f"- Average speedup: {df['speedup'].mean():.2f}x")
    print(f"- Best speedup: {df['speedup'].max():.2f}x ({df.loc[df['speedup'].idxmax(), 'kernel']})")
    print(f"- Peak GFLOPS: {df['best_gflops'].max():.1f} ({df.loc[df['best_gflops'].idxmax(), 'kernel']})")
    
    print("="*60)
    print("✓ Chart updated successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
