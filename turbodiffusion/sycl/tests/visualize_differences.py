#!/usr/bin/env python3
"""
Visualize differences between SYCL and reference outputs.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

def plot_error_distribution(ref, sycl, layer_name, save_path):
    """Plot error distribution histogram."""
    diff = np.abs(ref - sycl)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram of errors
    axes[0].hist(diff.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Absolute Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{layer_name} Error Distribution')
    axes[0].axvline(np.mean(diff), color='r', linestyle='--', label=f'Mean: {np.mean(diff):.2e}')
    axes[0].axvline(np.percentile(diff, 95), color='orange', linestyle='--', label=f'95th: {np.percentile(diff, 95):.2e}')
    axes[0].axvline(np.percentile(diff, 99), color='purple', linestyle='--', label=f'99th: {np.percentile(diff, 99):.2e}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Heatmap of errors (sample a subset)
    sample_size = min(100, diff.shape[0])
    im = axes[1].imshow(diff[:sample_size], aspect='auto', cmap='hot', interpolation='nearest')
    axes[1].set_xlabel('Feature Dimension')
    axes[1].set_ylabel('Sample Index')
    axes[1].set_title(f'Error Heatmap (Sample, n={sample_size})')
    plt.colorbar(im, ax=axes[1], label='Absolute Error')
    
    # Reference vs SYCL scatter
    sample_idx = np.random.choice(ref.size, size=min(10000, ref.size), replace=False)
    ref_flat = ref.flatten()
    sycl_flat = sycl.flatten()
    axes[2].scatter(ref_flat[sample_idx], sycl_flat[sample_idx], alpha=0.1, s=1, c='blue')
    min_val = min(ref.min(), sycl.min())
    max_val = max(ref.max(), sycl.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')
    axes[2].set_xlabel('Reference Value')
    axes[2].set_ylabel('SYCL Value')
    axes[2].set_title('Reference vs SYCL (10k samples)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")

def generate_all_visualizations(ref_outputs, sycl_outputs, output_dir='results'):
    """Generate all visualizations for all layers."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    print("-" * 60)
    
    for i, (ref, sycl) in enumerate(zip(ref_outputs, sycl_outputs)):
        layer_name = f'Block_{i}'
        save_path = os.path.join(output_dir, f'{layer_name.lower()}_error.png')
        plot_error_distribution(ref, sycl, layer_name, save_path)
    
    print(f"\nAll visualizations saved to {output_dir}/")

def plot_summary_statistics(ref_outputs, sycl_outputs, save_path='results/summary_stats.png'):
    """Plot summary statistics across all layers."""
    max_diffs = []
    mean_diffs = []
    p95_diffs = []
    p99_diffs = []
    
    for ref, sycl in zip(ref_outputs, sycl_outputs):
        diff = np.abs(ref - sycl)
        max_diffs.append(np.max(diff))
        mean_diffs.append(np.mean(diff))
        p95_diffs.append(np.percentile(diff, 95))
        p99_diffs.append(np.percentile(diff, 99))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Error metrics per layer
    x = np.arange(len(max_diffs))
    width = 0.2
    
    axes[0].bar(x - 1.5*width, max_diffs, width, label='Max', color='red', alpha=0.7)
    axes[0].bar(x - 0.5*width, mean_diffs, width, label='Mean', color='blue', alpha=0.7)
    axes[0].bar(x + 0.5*width, p95_diffs, width, label='95th percentile', color='orange', alpha=0.7)
    axes[0].bar(x + 1.5*width, p99_diffs, width, label='99th percentile', color='purple', alpha=0.7)
    
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Absolute Error')
    axes[0].set_title('Error Statistics per Layer')
    axes[0].set_xticks(x)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(1e-2, color='green', linestyle='--', linewidth=2, label='Tolerance (1e-2)')
    
    # Plot 2: Cumulative distribution of max errors
    sorted_max = np.sort(max_diffs)
    y = np.arange(1, len(sorted_max) + 1) / len(sorted_max)
    axes[1].plot(sorted_max, y, 'b-', linewidth=2, label='CDF of Max Errors')
    axes[1].axvline(1e-2, color='red', linestyle='--', linewidth=2, label='Tolerance (1e-2)')
    axes[1].set_xlabel('Max Absolute Error')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('CDF of Maximum Errors')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary statistics to {save_path}")

if __name__ == '__main__':
    # Example usage with dummy data
    print("Visualize Differences Tool")
    print("Usage: Import and use in validation script")
    print("\nExample:")
    print("  from visualize_differences import plot_error_distribution")
    print("  plot_error_distribution(ref_outputs[0], sycl_outputs[0], 'Block_0', 'block0_error.png')")
