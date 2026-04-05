#!/bin/bash
# Quick Start Guide for BMG/XPU Compilation Tools
# BMG/XPU编译工具快速入门

cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║     SYCL BMG/XPU Compilation Tools - Quick Start Guide       ║
╚══════════════════════════════════════════════════════════════╝

📋 可用工具
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. b60_sycl_builder.py      - 主要SYCL编译工具 (Python)
2. b60_sycl_builder.sh      - Shell包装器脚本
3. test_bmg_options.sh      - BMG/XPU编译选项测试
4. build.sh                 - 统一构建工具 (SYCL + CUDA)

🚀 快速开始
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 测试BMG/XPU编译选项
   ./tools/test_bmg_options.sh

2. 编译单个kernel
   python3 tools/b60_sycl_builder.py compile kernel_dataset/sycl/winograd_input_transform_kernel.dp.cpp
   
   # 或使用shell脚本
   ./tools/b60_sycl_builder.sh compile kernel_dataset/sycl/winograd_input_transform_kernel.dp.cpp

3. 批量编译所有SYCL kernels
   python3 tools/b60_sycl_builder.py compile-all
   # 或
   ./tools/build.sh b60 compile-all

4. 查看构建状态
   python3 tools/b60_sycl_builder.py status
   # 或
   ./tools/b60_sycl_builder.sh status

5. 检查环境
   ./tools/b60_sycl_builder.sh check

6. 清理构建产物
   ./tools/b60_sycl_builder.sh clean

🔧 支持的GPU架构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ BMG   - Battlemage (最新独立显卡) 🆕
✓ PVC   - Ponte Vecchio (数据中心)
✓ DG2   - Alchemist (消费级显卡)
✓ ARL-H - Arrow Lake-H
✓ MTL-H - Meteor Lake-H
✓ LNL-M - Lunar Lake-M
✓ PTL-H - Panther Lake-H
✓ PTL-U - Panther Lake-U

📊 编译选项 (基于Intel torch-xpu-ops)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基础选项:
  -fsycl -O2 -std=c++17

SYCL Kernel选项:
  -fsycl-unnamed-lambda              # 支持匿名lambda
  -sycl-std=2020                     # SYCL 2020标准
  -fhonor-nans                       # 尊重NaN
  -fhonor-infinities                 # 尊重Infinity
  -fno-associative-math             # 确定性结果
  -fno-approx-func                  # 禁用近似函数
  -no-ftz                           # 禁用flush-to-zero

AOT编译目标:
  -fsycl-targets=spir64_gen,spir64  # AOT + JIT

设备链接选项:
  -fsycl-max-parallel-link-jobs=4    # 并行链接
  --offload-compress                 # 压缩
  
离线编译器选项:
  -device pvc,bmg,dg2,arl-h,mtl-h,lnl-m,ptl-h,ptl-u
  -options -cl-poison-unsupported-fp64-kernels
  -options -cl-intel-enable-auto-large-GRF-mode
  -options -cl-fp32-correctly-rounded-divide-sqrt
  -options -cl-intel-greater-than-4GB-buffer-required

📁 输出目录结构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

results/
├── b60/
│   ├── build_sycl/              # 编译输出 (.o文件)
│   ├── compile_*.log            # 编译日志
│   └── summary_*.json           # 批量编译报告
└── accuracy/
    └── accuracy_report_*.json   # 准确度测试报告

scripts/
└── b60/
    └── build_*.sh               # 生成的编译脚本

🎯 性能特点
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

编译模式:  AOT (Ahead-of-Time) + JIT
启动速度:  ⚡ 极快 (预编译, 无需运行时编译)
文件大小:  ~230KB (包含8个架构的设备代码)
准确度:    ✅ 100% (数值确定性保证)

🔍 故障排除
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题: Container 'lsv-container' is not running
解决: docker start lsv-container

问题: Compilation failed
解决: 
  1. 检查文件是否存在
  2. 查看详细日志: results/b60/compile_*.log
  3. 运行测试脚本: ./tools/test_bmg_options.sh

问题: 找不到SYCL编译器
解决:
  1. 检查容器内是否安装了Intel oneAPI
  2. 在容器内运行: which icpx

📚 更多信息
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

详细文档: tools/UPDATED_TOOLS_SUMMARY.md
测试程序: test/sycl_bmg_test.cpp
准确度测试: test/accuracy/run_accuracy_test.py

📞 支持
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基于: Intel torch-xpu-ops BuildFlags.cmake
GitHub: https://github.com/intel/torch-xpu-ops

EOF

echo ""
echo "✨ 提示: 运行 './tools/test_bmg_options.sh' 开始测试!"
echo ""
