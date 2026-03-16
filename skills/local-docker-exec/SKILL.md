---
name: local-docker-exec
description: Execute commands in local Docker container (lsv-container) for compilation and testing. Use when building SYCL kernels locally on Intel GPU.
disable-model-invocation: true
allowed-tools: Bash
argument-hint: "[directory] [command] [output_file]"
---

# Local Docker Execution

Execute commands in the local Docker container (lsv-container) for local compilation and testing on Intel GPU.

## Configuration

- **Docker Container:** lsv-container
- **Host Base Directory:** /home/intel/tianfeng
- **Docker Workspace:** /intel/tianfeng/
- **Test Directory:** /home/intel/tianfeng/test

## Usage

```bash
/local-docker-exec <local_directory> [command] [output_file]
```

## Arguments

- `local_directory` - Local directory to copy and execute in (required)
- `command` - Command to run inside docker (default: bash)
- `output_file` - File to save output (default: auto-generated with timestamp)

## Examples

### Build SYCL tests locally
```bash
/local-docker-exec src/neural/backends/sycl "./build_test.sh --clean" build.log
```

### Run specific test
```bash
/local-docker-exec . "cd build && ./test_softmax" test.log
```

### Run all tests
```bash
/local-docker-exec src/neural/backends/sycl "./build_test.sh && cd build && ./test_softmax && ./test_layer_norm" all_tests.log
```

### Interactive bash session
```bash
/local-docker-exec src/neural/backends/sycl bash
```

## What This Skill Does

1. **Directory Check** - Verifies the local directory exists
2. **Local Copy** - Copies the directory to /home/intel/tianfeng/test/
3. **Container Check** - Verifies lsv-container is running (starts if stopped)
4. **Execution** - Runs the command in /intel/tianfeng/test/ inside the container
5. **Output Capture** - Collects stdout/stderr to the specified output file

## Local Execution Flow

```
Local Directory → Copy → /home/intel/tianfeng/test/ → Docker (lsv-container) → Execute → Output File
```

## Comparison with Remote Execution

| Feature | /local-docker-exec | /remote-docker-exec |
|---------|-------------------|---------------------|
| Target | Local lsv-container | Remote cuda12.9-test |
| Host | localhost | root@10.112.229.160 |
| GPU | Intel GPU | NVIDIA GPU |
| Use Case | Local SYCL dev/test | Remote CUDA testing |

## Troubleshooting

### Container Not Running
The script will try to start a stopped container automatically:
```bash
docker ps -a
docker start lsv-container
```

### Path Mapping
Remember the path mapping:
- Host path: /home/intel/tianfeng/test/<directory>
- Container path: /intel/tianfeng/test/<directory>

### Permission Issues
If you encounter permission issues in the container:
```bash
sudo chown -R $(id -u):$(id -g) /home/intel/tianfeng/test/
```

## Implementation

This skill invokes: `./local_docker_exec.sh $ARGUMENTS`
