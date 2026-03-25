#!/usr/bin/env python3
"""
Comprehensive Benchmark Analysis for All 23 Kernels
Generates visualization charts for the complete kernel test suite.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Kernel categories with their result files
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

def parse_csv(filepath):
    """Parse a kernel result CSV file."""
    try:
        # Read CSV, skip header comments
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find header line (starts with Version)
        header_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('Version') or 'Version' in line:
                header_idx = i
                break
        
        # Parse data
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

def extract_peak_performance(df, kernel_name):
    """Extract peak performance metrics from a dataframe."""
    if df is None or df.empty:
        return None
    
    result = {'kernel': kernel_name}
    
    # Find best version by GFLOPS or Bandwidth
    if 'GFLOPS' in df.columns:
        df['GFLOPS'] = pd.to_numeric(df['GFLOPS'], errors='coerce')
        best_idx = df['GFLOPS'].idxmax()
        result['peak_gflops'] = df.loc[best_idx, 'GFLOPS']
        result['best_version'] = df.loc[best_idx, 'Version']
        result['metric'] = 'GFLOPS'
    elif 'Bandwidth_GB/s' in df.columns:
        df['Bandwidth_GB/s'] = pd.to_numeric(df['Bandwidth_GB/s'], errors='coerce')
        best_idx = df['Bandwidth_GB/s'].idxmax()
        result['peak_bandwidth'] = df.loc[best_idx, 'Bandwidth_GB/s']
        result['best_version'] = df.loc[best_idx, 'Version']
        result['metric'] = 'Bandwidth'
    
    # Calculate speedup if multiple versions
    if 'GFLOPS' in df.columns and df['Version'].nunique() > 1:
        v0_gflops = df[df['Version'] == 'V0']['GFLOPS'].max()
        best_gflops = df['GFLOPS'].max()
        if v0_gflops > 0:
            result['speedup'] = best_gflops / v0_gflops
    
    return result

def create_performance_overview():
    """Chart 1: Performance overview of all kernels."""
    results_dir = Path('/home/intel/tianfeng/opencode_bench/benchmarks/results')
    
    all_kernels = []
    for category, kernels in KERNELS.items():
        for kernel in kernels:
            filepath = results_dir / f"{kernel}_results.csv"
            df = parse_csv(filepath)
            perf = extract_peak_performance(df, kernel)
            if perf:
                perf['category'] = category
                all_kernels.append(perf)
    
    df_summary = pd.DataFrame(all_kernels)
    
    # Separate GFLOPS and Bandwidth kernels
    gflops_kernels = df_summary[df_summary['metric'] == 'GFLOPS'].copy()
    bandwidth_kernels = df_summary[df_summary['metric'] == 'Bandwidth'].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: GFLOPS
    if not gflops_kernels.empty:
        gflops_kernels = gflops_kernels.sort_values('peak_gflops', ascending=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(gflops_kernels)))
        
        bars = ax1.barh(gflops_kernels['kernel'], gflops_kernels['peak_gflops'], color=colors)
        ax1.set_xlabel('Peak Performance (GFLOPS)', fontsize=12, fontweight='bold')
        ax1.set_title('All 23 Kernels - Peak GFLOPS Performance\n(Intel Graphics [0xe211] - Battlemage G21)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(gflops_kernels.iterrows()):
            ax1.text(row['peak_gflops'] + 10, i, f"{row['peak_gflops']:.1f}", 
                    va='center', fontsize=9)
    
    # Plot 2: Bandwidth
    if not bandwidth_kernels.empty:
        bandwidth_kernels = bandwidth_kernels.sort_values('peak_bandwidth', ascending=True)
        colors = plt.cm.plasma(np.linspace(0, 1, len(bandwidth_kernels)))
        
        bars = ax2.barh(bandwidth_kernels['kernel'], bandwidth_kernels['peak_bandwidth'], color=colors)
        ax2.set_xlabel('Peak Bandwidth (GB/s)', fontsize=12, fontweight='bold')
        ax2.set_title('Memory-Bound Kernels - Peak Bandwidth', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(bandwidth_kernels.iterrows()):
            ax2.text(row['peak_bandwidth'] + 5, i, f"{row['peak_bandwidth']:.1f}", 
                    va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/intel/tianfeng/opencode_bench/benchmarks/charts/all_kernels_performance.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Created: all_kernels_performance.png")
    plt.close()
    
    return df_summary

def create_category_comparison(df_summary):
    """Chart 2: Performance by category."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # GFLOPS by category
    gflops_by_cat = df_summary[df_summary['metric'] == 'GFLOPS'].groupby('category')['peak_gflops'].agg(['mean', 'max', 'count'])
    
    if not gflops_by_cat.empty:
        x = np.arange(len(gflops_by_cat))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, gflops_by_cat['mean'], width, label='Average', alpha=0.8)
        bars2 = ax1.bar(x + width/2, gflops_by_cat['max'], width, label='Peak', alpha=0.8)
        
        ax1.set_xlabel('Kernel Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('GFLOPS', fontsize=12, fontweight='bold')
        ax1.set_title('Performance by Category (GFLOPS)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(gflops_by_cat.index, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # Count by category
    cat_counts = df_summary.groupby('category').size()
    colors = plt.cm.Set3(np.linspace(0, 1, len(cat_counts)))
    
    wedges, texts, autotexts = ax2.pie(cat_counts.values, labels=cat_counts.index, autopct='%1.0f%%',
                                        colors=colors, startangle=90)
    ax2.set_title('Distribution of Kernels by Category\n(Total: 23 Kernels)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/intel/tianfeng/opencode_bench/benchmarks/charts/category_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Created: category_analysis.png")
    plt.close()

def create_version_comparison():
    """Chart 3: V0 vs V1 vs V2 comparison across all kernels."""
    results_dir = Path('/home/intel/tianfeng/opencode_bench/benchmarks/results')
    
    version_data = []
    
    for category, kernels in KERNELS.items():
        for kernel in kernels:
            filepath = results_dir / f"{kernel}_results.csv"
            df = parse_csv(filepath)
            
            if df is not None and 'GFLOPS' in df.columns:
                df['GFLOPS'] = pd.to_numeric(df['GFLOPS'], errors='coerce')
                
                for version in ['V0', 'V1', 'V2']:
                    version_df = df[df['Version'] == version]
                    if not version_df.empty:
                        version_data.append({
                            'kernel': kernel,
                            'category': category,
                            'version': version,
                            'avg_gflops': version_df['GFLOPS'].mean(),
                            'max_gflops': version_df['GFLOPS'].max()
                        })
    
    df_versions = pd.DataFrame(version_data)
    
    if df_versions.empty:
        print("⚠ No version data available")
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Pivot for plotting
    pivot_avg = df_versions.pivot_table(values='avg_gflops', index='kernel', columns='version', fill_value=0)
    
    # Select top 15 kernels by V0 performance for clarity
    if len(pivot_avg) > 15:
        top_kernels = pivot_avg['V0'].nlargest(15).index
        pivot_avg = pivot_avg.loc[top_kernels]
    
    x = np.arange(len(pivot_avg))
    width = 0.25
    
    bars0 = ax.bar(x - width, pivot_avg.get('V0', 0), width, label='V0 (Baseline)', alpha=0.8, color='#e74c3c')
    bars1 = ax.bar(x, pivot_avg.get('V1', 0), width, label='V1 (Optimized)', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width, pivot_avg.get('V2', 0), width, label='V2 (Alternative)', alpha=0.8, color='#2ecc71')
    
    ax.set_xlabel('Kernel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average GFLOPS', fontsize=12, fontweight='bold')
    ax.set_title('Version Comparison Across All Kernels (Top 15 by Performance)\nV0 vs V1 vs V2', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_avg.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/intel/tianfeng/opencode_bench/benchmarks/charts/version_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Created: version_comparison.png")
    plt.close()

def create_optimization_insights(df_summary):
    """Chart 4: Optimization insights and speedup analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Speedup distribution
    speedup_data = df_summary[df_summary['speedup'].notna()]['speedup']
    if not speedup_data.empty:
        ax1.hist(speedup_data, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(speedup_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {speedup_data.mean():.2f}x')
        ax1.set_xlabel('Speedup Factor (Best vs Baseline)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Kernels', fontsize=11, fontweight='bold')
        ax1.set_title('Distribution of Optimization Speedup', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
    # 2. Best version distribution
    if 'best_version' in df_summary.columns:
        version_counts = df_summary['best_version'].value_counts()
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        ax2.pie(version_counts.values, labels=version_counts.index, autopct='%1.0f%%',
               colors=colors[:len(version_counts)], startangle=90)
        ax2.set_title('Best Performing Version Distribution', fontsize=13, fontweight='bold')
    
    # 3. Top performers
    gflops_kernels = df_summary[df_summary['metric'] == 'GFLOPS'].nlargest(10, 'peak_gflops')
    if not gflops_kernels.empty:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(gflops_kernels)))
        bars = ax3.barh(gflops_kernels['kernel'], gflops_kernels['peak_gflops'], color=colors)
        ax3.set_xlabel('Peak GFLOPS', fontsize=11, fontweight='bold')
        ax3.set_title('Top 10 Performing Kernels (GFLOPS)', fontsize=13, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
    
    # 4. Category performance summary
    if not df_summary.empty:
        cat_perf = df_summary.groupby('category').apply(
            lambda x: x.loc[x['peak_gflops'].idxmax()] if 'peak_gflops' in x.columns and x['peak_gflops'].notna().any() 
            else x.iloc[0]
        )
        
        if 'peak_gflops' in cat_perf.columns:
            cat_perf = cat_perf.sort_values('peak_gflops', ascending=True)
            colors = plt.cm.plasma(np.linspace(0, 1, len(cat_perf)))
            bars = ax4.barh(cat_perf.index, cat_perf['peak_gflops'], color=colors)
            ax4.set_xlabel('Peak GFLOPS (Best in Category)', fontsize=11, fontweight='bold')
            ax4.set_title('Peak Performance by Category', fontsize=13, fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/intel/tianfeng/opencode_bench/benchmarks/charts/optimization_insights.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Created: optimization_insights.png")
    plt.close()

def create_workgroup_analysis():
    """Chart 5: Work-group size impact analysis."""
    # Manual data based on test results
    wg_data = {
        'Element-wise': {'128': 9, '256': 1, 'tested': 10},
        'Reduction': {'128': 1, '256': 2, 'tested': 3},
        'Spatial': {'256': 3, 'tested': 3},  # 3D configurations
        'Filter/Matrix': {'256': 2, 'tested': 2},
        'Complex': {'128': 1, 'single': 1, 'tested': 2}
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: WG size distribution by category
    categories = list(wg_data.keys())
    wg_128 = [wg_data[cat].get('128', 0) for cat in categories]
    wg_256 = [wg_data[cat].get('256', 0) for cat in categories]
    wg_other = [wg_data[cat].get('single', 0) + wg_data[cat].get('64', 0) for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.6
    
    p1 = ax1.bar(x, wg_128, width, label='WG=128', color='#3498db', alpha=0.8)
    p2 = ax1.bar(x, wg_256, width, bottom=wg_128, label='WG=256', color='#e74c3c', alpha=0.8)
    p3 = ax1.bar(x, wg_other, width, bottom=np.array(wg_128) + np.array(wg_256), 
                label='Other', color='#95a5a6', alpha=0.8)
    
    ax1.set_ylabel('Number of Kernels', fontsize=12, fontweight='bold')
    ax1.set_title('Optimal Work-Group Size by Kernel Category', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: WG size recommendation pie
    wg_totals = {
        'WG=128 (Optimal)': 10,
        'WG=256 (Optimal)': 8,
        'Other/Mixed': 5
    }
    
    colors = ['#3498db', '#e74c3c', '#95a5a6']
    wedges, texts, autotexts = ax2.pie(wg_totals.values(), labels=wg_totals.keys(), 
                                        autopct='%1.0f%%', colors=colors, startangle=90)
    ax2.set_title('Work-Group Size Distribution\n(23 Kernels)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/intel/tianfeng/opencode_bench/benchmarks/charts/workgroup_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Created: workgroup_analysis.png")
    plt.close()

def main():
    """Generate all charts."""
    print("="*60)
    print("Generating Comprehensive Benchmark Charts")
    print("All 23 Kernels Analysis")
    print("="*60)
    
    # Create summary dataframe
    df_summary = create_performance_overview()
    
    if df_summary is not None and not df_summary.empty:
        create_category_comparison(df_summary)
        create_optimization_insights(df_summary)
    
    create_version_comparison()
    create_workgroup_analysis()
    
    print("="*60)
    print("✓ All charts generated successfully!")
    print("Location: benchmarks/charts/")
    print("="*60)

if __name__ == "__main__":
    main()
