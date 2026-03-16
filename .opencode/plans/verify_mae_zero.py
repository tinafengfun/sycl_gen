#!/usr/bin/env python3
"""
验证 MAE=0 内核的真实性
检查测试是否真正执行了内核功能，而不是假阳性
"""

import sys
sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/.opencode/plans')

from FINAL_ALL_HARNESSES import ALL_HARNESSES

def analyze_kernel(kernel_id, harness):
    """分析内核测试是否正确"""
    issues = []
    
    cuda_code = harness['cuda']
    sycl_code = harness['sycl']
    
    # 检查 CUDA 代码
    cuda_checks = {
        'has_kernel': '__global__' in cuda_code or 'template' in cuda_code,
        'has_main': 'int main()' in cuda_code,
        'launches_kernel': '<<<>>>' in cuda_code,
        'allocates_memory': 'cudaMalloc' in cuda_code,
        'copies_data': 'cudaMemcpy' in cuda_code
    }
    
    # 检查 SYCL 代码
    sycl_checks = {
        'has_queue': 'sycl::queue' in sycl_code,
        'has_main': 'int main()' in sycl_code,
        'uses_parallel_for': 'parallel_for' in sycl_code,
        'allocates_memory': 'malloc_device' in sycl_code,
        'copies_data': 'memcpy' in sycl_code
    }
    
    # 检查逻辑匹配
    if cuda_checks['has_kernel'] and sycl_checks['uses_parallel_for']:
        # 检查是否都是真正的计算，不是简单的memcpy
        cuda_has_compute = any(op in cuda_code for op in ['+', '-', '*', '/', 'exp', 'log', 'sqrt', 'sin', 'cos'])
        sycl_has_compute = any(op in sycl_code for op in ['+', '-', '*', '/', 'exp', 'log', 'sqrt', 'sin', 'cos'])
        
        if not cuda_has_compute or not sycl_has_compute:
            issues.append("WARNING: May be simple memory copy without actual computation")
    
    return {
        'kernel_id': kernel_id,
        'cuda_checks': cuda_checks,
        'sycl_checks': sycl_checks,
        'issues': issues,
        'valid': all(cuda_checks.values()) and all(sycl_checks.values()) and len(issues) == 0
    }

def main():
    print("=" * 80)
    print("🔍 验证 MAE=0 内核的真实性")
    print("=" * 80)
    print()
    
    # 重点关注之前显示 MAE=0 的内核
    mae_zero_kernels = [
        'copy_type_converted',
        'expand_planes_nchw', 
        'expand_planes_nhwc',
        'global_scale_fp16_nhwc',
        'gen_offset_pointers',
        'winograd_filter_transform',
        'winograd_output_transform',
        'winograd_output_se_relu_input',
        'winograd_output_relu_input',
        'output_input_transform_fp16_shmem',
        'add_vectors',
        'winograd_input_transform',
        'add_vectors_hnc_nhc',
        'add_bias_nchw',
        'nchw_to_nhwc',
        'add_bias_batched',
        'global_scale'
    ]
    
    results = []
    
    print("分析内核实现质量：\n")
    
    for kernel_id in mae_zero_kernels:
        if kernel_id not in ALL_HARNESSES:
            print(f"⚠️  {kernel_id}: 未找到")
            continue
            
        harness = ALL_HARNESSES[kernel_id]
        analysis = analyze_kernel(kernel_id, harness)
        results.append(analysis)
        
        status = "✅" if analysis['valid'] else "❌"
        print(f"{status} {kernel_id}")
        
        if analysis['issues']:
            for issue in analysis['issues']:
                print(f"   ⚠️  {issue}")
    
    print("\n" + "=" * 80)
    print("📊 验证总结")
    print("=" * 80)
    
    valid_count = sum(1 for r in results if r['valid'])
    total = len(results)
    
    print(f"\n✅ 有效实现: {valid_count}/{total}")
    print(f"❌ 存在问题: {total - valid_count}/{total}")
    
    if total - valid_count > 0:
        print("\n需要修复的内核：")
        for r in results:
            if not r['valid']:
                print(f"  - {r['kernel_id']}: {', '.join(r['issues'])}")
    
    print("\n" + "=" * 80)
    print("🔍 深度检查：随机选取3个内核验证输出逻辑")
    print("=" * 80)
    
    # 深度检查几个关键内核
    check_kernels = ['copy_type_converted', 'expand_planes_nchw', 'add_vectors']
    
    for kernel_id in check_kernels:
        if kernel_id in ALL_HARNESSES:
            print(f"\n{kernel_id}:")
            harness = ALL_HARNESSES[kernel_id]
            
            # 检查 CUDA kernel 是否有实际计算
            cuda_kernel_start = harness['cuda'].find('__global__')
            cuda_kernel_end = harness['cuda'].find('}', cuda_kernel_start)
            if cuda_kernel_start > 0 and cuda_kernel_end > 0:
                kernel_body = harness['cuda'][cuda_kernel_start:cuda_kernel_end]
                ops = []
                if '+' in kernel_body: ops.append('加法')
                if '-' in kernel_body: ops.append('减法')
                if '*' in kernel_body: ops.append('乘法')
                if '/' in kernel_body: ops.append('除法')
                if 'exp' in kernel_body: ops.append('指数')
                if 'sqrt' in kernel_body: ops.append('开方')
                if 'half' in kernel_body or '__float2half' in kernel_body: ops.append('类型转换')
                
                print(f"  CUDA Kernel 操作: {', '.join(ops) if ops else '简单复制'}")
            
            # 检查 SYCL 是否有实际计算
            sycl_parallel_start = harness['sycl'].find('parallel_for')
            if sycl_parallel_start > 0:
                parallel_body = harness['sycl'][sycl_parallel_start:sycl_parallel_start+500]
                ops = []
                if '+' in parallel_body: ops.append('加法')
                if '-' in parallel_body: ops.append('减法')
                if '*' in parallel_body: ops.append('乘法')
                if '/' in parallel_body: ops.append('除法')
                if 'exp' in parallel_body: ops.append('指数')
                if 'sqrt' in parallel_body: ops.append('开方')
                if 'half' in parallel_body: ops.append('类型转换')
                
                print(f"  SYCL Kernel 操作: {', '.join(ops) if ops else '简单复制'}")

if __name__ == '__main__':
    main()
