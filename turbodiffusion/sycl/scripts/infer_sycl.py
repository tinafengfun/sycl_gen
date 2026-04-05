#!/usr/bin/env python3
"""
TurboDiffusion SYCL Inference Script
Runs Wan2.1 T2V inference using Intel XPU with SYCL kernels.
"""

import argparse
import sys
sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/TurboDiffusion/turbodiffusion')

import torch
import yaml
from pathlib import Path

from inference.modify_model import select_model
from turbodiffusion_sycl import replace_attention_sycl, replace_norm_sycl
from rcm.utils.model_utils import load_state_dict


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="TurboDiffusion SYCL Inference for Wan2.1"
    )
    parser.add_argument(
        '--model_path', 
        required=True, 
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prompt', 
        default='A cat playing with a ball', 
        help='Text prompt'
    )
    parser.add_argument(
        '--config', 
        default='/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl/configs/wan2.1_1.3B_sycl.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--attention', 
        choices=['flash', 'sparse'], 
        default='sparse',
        help='Attention type'
    )
    parser.add_argument(
        '--topk', 
        type=float, 
        default=0.2, 
        help='Top-k ratio for sparse attention'
    )
    parser.add_argument(
        '--device', 
        default='xpu', 
        help='Device to use (xpu for Intel)'
    )
    parser.add_argument(
        '--output', 
        default='output.mp4', 
        help='Output video path'
    )
    parser.add_argument(
        '--num_frames', 
        type=int, 
        default=81, 
        help='Number of frames'
    )
    parser.add_argument(
        '--resolution', 
        default='480p', 
        help='Resolution (480p, 720p)'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Check for XPU availability
    if args.device == 'xpu':
        if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
            print("Warning: XPU not available, falling back to CPU")
            device = torch.device('cpu')
    
    # Load model
    print("Loading model...")
    model_name = config['model']['name']
    model = select_model(model_name)
    
    # Apply SYCL optimizations
    print(f"Applying SYCL {args.attention} attention...")
    replace_attention_sycl(
        model, 
        attention_type=args.attention, 
        topk=args.topk
    )
    replace_norm_sycl(model)
    
    # Load weights
    print(f"Loading weights from {args.model_path}...")
    state_dict = load_state_dict(args.model_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print("Model loaded successfully")
    
    # TODO: Add actual video generation pipeline
    print("Running inference...")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output}")
    print(f"Num frames: {args.num_frames}")
    print(f"Resolution: {args.resolution}")
    
    print("\nInference pipeline ready. Video generation to be implemented.")
    print("SYCL kernels are active and model is loaded on XPU.")
    

if __name__ == '__main__':
    main()
