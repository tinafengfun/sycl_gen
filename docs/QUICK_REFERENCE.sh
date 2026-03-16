#!/bin/bash
# Quick Reference Card for opencode skill usage
# 在opencode中使用skill的快速参考

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════╗
║           opencode Skill 快速使用指南                              ║
╚══════════════════════════════════════════════════════════════════╝

📁 目录结构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
.opencode/
├── config.json          # opencode配置
└── skills/
    ├── b60-sycl-builder/SKILL.md      # SYCL编译skill
    └── remote-cuda-builder/SKILL.md   # CUDA编译skill

🔧 环境要求
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Docker环境: lsv-container (B60)
2. 远程CUDA: cuda12.9-test@10.112.229.160
3. SSH免密登录已配置

🚀 快速开始
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 检查环境
   $ ./tools/test_connectivity.sh

2. 编译单个Kernel
   # SYCL
   $ ./tools/build.sh b60 compile kernel_dataset/sycl/xxx.dp.cpp
   
   # CUDA
   $ ./tools/build.sh cuda compile kernel_dataset/cuda/xxx.cu

3. 批量编译
   $ ./tools/build.sh b60 compile-all
   $ ./tools/build.sh cuda compile-all

4. 查看状态
   $ ./tools/build.sh all status

💬 在opencode对话中使用
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

方式1: 直接请求（Agent自动使用skill）
─────────────────────────────────────────
👉 "编译 kernel_dataset/sycl/add_vectors.dp.cpp"
👉 "使用B60环境编译所有SYCL kernel"
👉 "检查编译状态"
👉 "在远程CUDA上编译测试kernel"

方式2: 显式调用工具
─────────────────────────────────────────
👉 "运行 ./tools/build.sh b60 compile-all"
👉 "执行 ./tools/test_connectivity.sh"
👉 "查看 results/b60/compile_*.log"

方式3: 批量操作
─────────────────────────────────────────
👉 "批量编译所有未编译的kernel"
👉 "对比CUDA和SYCL的编译结果"
👉 "清理所有构建产物"

📊 输出文件
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

results/
├── b60/                              # SYCL编译结果
│   ├── build_sycl/*.o               # 编译产物
│   ├── compile_*_YYYYMMDD_HHMMSS.log # 编译日志
│   └── summary_YYYYMMDD_HHMMSS.json  # 汇总报告
│
└── cuda/                             # CUDA编译结果
    ├── build_cuda/*.o               # 编译产物
    ├── compile_*_YYYYMMDD_HHMMSS.log # 编译日志
    └── summary_YYYYMMDD_HHMMSS.json  # 汇总报告

scripts/
├── b60/build_*_*.sh                 # 生成的SYCL编译脚本
└── cuda/build_*_*.sh                # 生成的CUDA编译脚本

.build_status.json                   # 构建状态追踪

🔍 故障排除
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题1: 容器未运行
  解决: docker start lsv-container

问题2: SSH连接失败
  解决: ssh-copy-id root@10.112.229.160

问题3: 编译失败
  解决: 查看 results/{env}/compile_*.log

问题4: Skill未加载
  解决: /reload 或重启opencode

📚 详细文档
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 集成指南: docs/OPENCODE_INTEGRATION.md
- 工具文档: tools/README.md
- 测试报告: test/TEST_REPORT.md
- Skill文档: .opencode/skills/*/SKILL.md

🎯 常用命令速查
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 环境
./tools/test_connectivity.sh         # 测试环境连通性
./tools/build.sh b60 check           # 检查B60环境
./tools/build.sh cuda check          # 检查CUDA环境

# 编译
./tools/build.sh b60 compile <file>     # 编译SYCL
./tools/build.sh cuda compile <file>    # 编译CUDA
./tools/build.sh all compile-all        # 编译全部

# 状态
./tools/build.sh b60 status          # SYCL状态
./tools/build.sh cuda status         # CUDA状态
./tools/build.sh all status          # 全部状态

# 清理
./tools/build.sh b60 clean           # 清理SYCL
./tools/build.sh cuda clean          # 清理CUDA
./tools/build.sh all clean           # 清理全部

# 日志
cat results/b60/compile_*.log        # 查看SYCL日志
cat results/cuda/compile_*.log       # 查看CUDA日志

╔══════════════════════════════════════════════════════════════════╗
║  提示: 将此文件作为quick reference，在opencode对话中直接参考使用   ║
╚══════════════════════════════════════════════════════════════════╝
EOF
