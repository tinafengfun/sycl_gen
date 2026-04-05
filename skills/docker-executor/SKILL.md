---
name: docker-executor
description: Execute commands in Docker containers for GPU kernel compilation and testing. Supports local (Intel GPU) and remote (NVIDIA GPU) execution modes.
disable-model-invocation: true
allowed-tools: Bash
argument-hint: "[mode] [directory] [command] [output_file]"
---

# Docker Executor

Unified Docker execution for GPU kernel development. Supports local and remote containers.

## Modes

| Mode | Container | Host | GPU | Use Case |
|------|-----------|------|-----|----------|
| `local` | lsv-container | localhost | Intel GPU | SYCL build/test |
| `remote` | cuda12.9-test | root@10.112.229.160 | NVIDIA GPU | CUDA build/test |
| `debug` | lsv-container | localhost | Intel GPU | SYCL debugging |

## Usage

```bash
/docker-executor <mode> <directory> [command] [output_file]
```

### Local SYCL Build & Test
```bash
/docker-executor local kernel_bench/tests "icpx -fsycl -O2 test_softmax_r1.cpp -o test_softmax && ./test_softmax"
```

### Remote CUDA Build & Test
```bash
/docker-executor remote kernel_bench/kernel_dataset/cuda "nvcc -O2 softmax_kernel.cu -o test_softmax && ./test_softmax"
```

### SYCL Debug Session
```bash
/docker-executor debug kernel_bench/tests bash
```

## Configuration

### Local Mode
- **Container:** lsv-container
- **Host Base Dir:** /home/intel/tianfeng
- **Container Workspace:** /intel/tianfeng/
- **Path Mapping:** Host /home/intel/tianfeng → Container /intel/tianfeng

### Remote Mode
- **Host:** root@10.112.229.160
- **Container:** cuda12.9-test
- **Path Mapping:** Host /home/tianfeng → Container /workspace

## What This Skill Does

1. **Directory Check** - Verifies the local directory exists
2. **Transfer** - Copies directory to target (local or remote)
3. **Container Check** - Verifies container is running (starts if stopped)
4. **Execution** - Runs command inside container
5. **Output Capture** - Collects stdout/stderr to output file

## Troubleshooting

### Container Not Running
```bash
# Local
docker ps -a && docker start lsv-container

# Remote
ssh root@10.112.229.160 "docker start cuda12.9-test"
```

### SSH Connection (remote mode)
```bash
ssh-copy-id root@10.112.229.160
```

## Implementation

- Local mode: uses `local_docker_exec.sh`
- Remote mode: uses `remote_docker_exec.sh`
- Python alternative: `local_docker_exec.py` (advanced features)
