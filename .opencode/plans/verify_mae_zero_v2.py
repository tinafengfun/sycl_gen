#!/usr/bin/env python3
"""
验证 MAE=0 内核的真实性 - 修正版
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
        'has_kernel_or_template': '__global__' in cuda_code or 'template' in cuda_code,
        'has_main': 'int main()' in cuda_code,
        'launches_kernel': '<<<' in cuda_code,
        'allocates_memory': 'cudaMalloc' in cuda_code or 'new' in cuda_code,
        'copies_data': 'cudaMemcpy' in cuda_code or 'memcpy' in cuda_code
    }
    
    # 检查 SYCL 代码
    sycl_checks = {
        'has_queue': 'sycl::queue' in sycl_code,
        'has_main': 'int main()' in sycl_code,
        'uses_parallel_for': 'parallel_for' in sycl_code,
        'allocates_memory': 'malloc_device' in sycl_code or 'new' in sycl_code,
        'copies_data': 'memcpy' in sycl_code
    }
    
    # 关键：检查是否实际执行了计算（不是简单的 memcpy）
    # 检查 kernel 体内是否有计算
    cuda_kernel_body = ""
    sycl_kernel_body = ""
    
    # 提取 CUDA kernel body
    if '__global__' in cuda_code:
        kernel_start = cuda_code.find('__global__')
        brace_start = cuda_code.find('{', kernel_start)
        # 找到匹配的 }
        brace_count = 0
        kernel_end = brace_start
        for i, c in enumerate(cuda_code[brace_start:]):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    kernel_end = brace_start + i
                    break
        cuda_kernel_body = cuda_code[brace_start:kernel_end+1]
    
    # 提取 SYCL kernel body
    if 'parallel_for' in sycl_code:
        pf_start = sycl_code.find('parallel_for')
        lambda_start = sycl_code.find('{', pf_start)
        brace_count = 0
        lambda_end = lambda_start
        for i, c in enumerate(sycl_code[lambda_start:]):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    lambda_end = lambda_start + i
                    break
        sycl_kernel_body = sycl_code[lambda_start:lambda_end+1]
    
    # 检查是否有实际计算（排除内存操作）
    compute_ops = ['+', '-', '*', '/', '%', '<<', '>>', '&', '|', '^', '~']
    math_ops = ['exp', 'log', 'sqrt', 'sin', 'cos', 'tanh', 'pow', 'fmax', 'fmin', 'max', 'min']
    
    cuda_has_compute = any(op in cuda_kernel_body for op in compute_ops) or \
                       any(op in cuda_kernel_body for op in math_ops)
    sycl_has_compute = any(op in sycl_kernel_body for op in compute_ops) or \
                       any(op in sycl_kernel_body for op in math_ops)
    
    # 特殊处理：类型转换也算计算
    if 'half' in cuda_kernel_body or '__float2half' in cuda_kernel_body:
        cuda_has_compute = True
    if 'sycl::half' in sycl_kernel_body or 'half' in sycl_kernel_body:
        sycl_has_compute = True
    
    if not cuda_has_compute:
        issues.append("CUDA kernel 可能没有实际计算 (只是内存操作)")
    if not sycl_has_compute:
        issues.append("SYCL kernel 可能没有实际计算 (只是内存操作)")
    
    # 检查输入输出是否不同（防止 d_output[i] = d_input[i] 这样的简单复制）
    # 这是一个简单的启发式检查
    if cuda_kernel_body.count('=') == 1 and 'd_output' in cuda_kernel_body and 'd_input' in cuda_kernel_body:
        # 只有一个赋值，可能是简单复制
        if '+' not in cuda_kernel_body and '*' not in cuda_kernel_body:
            issues.append("CUDA 可能是简单 memcpy")
    
    valid = all(cuda_checks.values()) and all(sycl_checks.values()) and cuda_has_compute and sycl_has_compute
    
    return {
        'kernel_id': kernel_id,
        'cuda_checks': cuda_checks,
        'sycl_checks': sycl_checks,
        'cuda_has_compute': cuda_has_compute,
        'sycl_has_compute': sycl_has_compute,
        'issues': issues,
        'valid': valid
    }

def main():
    print("=" * 80)
    print("🔍 验证 MAE=0 内核的真实性 - 深度分析")
    print("=" * 80)
    print()
    
    # 所有显示 MAE=0 的内核
    mae_zero_kernels = [
        'copy_type_converted',
        'expand_planes_nchw', 
        'expand_planes_nhwc',
        'global_scale_fp16_nhwc',
        'gen_offset_pointers',
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
    print(f"{'内核名称':<30} {'CUDA计算':<10} {'SYCL计算':<10} {'状态':<10} {'问题'}")
    print("-" * 100)
    
    for kernel_id in mae_zero_kernels:
        if kernel_id not in ALL_HARNESSES:
            print(f"{kernel_id:<30} {'N/A':<10} {'N/A':<10} {'❌ 未找到':<10}")
            continue
            
        harness = ALL_HARNESSES[kernel_id]
        analysis = analyze_kernel(kernel_id, harness)
        results.append(analysis)
        
        cuda_comp = "✅" if analysis['cuda_has_compute'] else "❌"
        sycl_comp = "✅" if analysis['sycl_has_compute'] else "❌"
        status = "✅ 有效" if analysis['valid'] else "❌ 可疑"
        issues = "; ".join(analysis['issues']) if analysis['issues'] else "无"
        
        print(f"{kernel_id:<30} {cuda_comp:<10} {sycl_comp:<10} {status:<10} {issues}")
    
    print("\n" + "=" * 80)
    print("📊 验证总结")
    print("=" * 80)
    
    valid_count = sum(1 for r in results if r['valid'])
    suspicious_count = len([r for r in results if r['issues'] and not r['valid']])
    total = len([r for r in results if r['kernel_id'] in ALL_HARNESSES])
    
    print(f"\n✅ 有效实现: {valid_count}/{total}")
    print(f"⚠️  可疑实现: {suspicious_count}/{total}")
    print(f"❌ 未找到: {len(mae_zero_kernels) - total}/{len(mae_zero_kernels)}")
    
    if suspicious_count > 0:
        print("\n⚠️  可疑实现详情：")
        for r in results:
            if r['issues']:
                print(f"\n  {r['kernel_id']}:")
                for issue in r['issues']:
                    print(f"    - {issue}")
    
    # 检查关键问题：MAE=0 是否合理？
    print("\n" + "=" * 80)
    print("🤔 为什么 MAE=0？")
    print("=" * 80)
    print("""
MAE=0 表示 CUDA 和 SYCL 输出完全相同。这在以下情况下是合理的：

✅ 合理情况：
1. 简单的类型转换（如 float→half），两种实现使用相同精度的算法
2. 确定性操作（如 bit 操作、索引计算），无浮点误差
3. 整数运算或位运算

⚠️ 可疑情况：
1. 浮点运算应该有小误差，但 MAE=0 表示可能用了相同的主机端实现
2. SYCL 代码实际上在主机端执行（没有用 parallel_for 或用了错误的内存访问）
3. 两个输出都是 0 或未初始化内存

需要手动验证：
- 运行单个测试，检查输出文件是否包含预期的非零值
- 确保 SYCL 代码真正在 GPU 上执行（检查 sycl::gpu_selector_v）
- 验证输出值范围是否合理（不是全 0 或随机值）
""")

if __name__ == '__main__':
    main()
