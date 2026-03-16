---
name: remote-docker-exec
description: Execute commands in remote Docker container for compilation and testing. Use when building SYCL kernels, running tests on Intel GPU, or executing remote compile jobs.
disable-model-invocation: true
allowed-tools: Bash
argument-hint: "[directory] [command] [output_file]"
---

# Remote Docker Execution

Execute commands in a remote Docker container (cuda12.9-test on root@10.112.229.160) for remote compilation and testing.

## Configuration

- **Remote Host:** root@10.112.229.160
- **Docker Container:** cuda12.9-test
- **Code Mapping:** /home/tianfeng (host) → /workspace (docker)

## Usage

```bash
/remote-docker-exec <local_directory> [command] [output_file]
```

## Arguments

- `local_directory` - Local directory to copy and execute in (required)
- `command` - Command to run inside docker (default: bash)
- `output_file` - File to save output (default: auto-generated with timestamp)

## Examples

### Build SYCL tests remotely
```bash
/remote-docker-exec src/neural/backends/sycl "./build_test.sh --clean" build.log
```

### Run specific test
```bash
/remote-docker-exec . "cd build && ./test_softmax" test.log
```

### Run all tests with parsing
```bash
/remote-docker-exec src/neural/backends/sycl "./build_test.sh && cd build && ./test_softmax && ./test_layer_norm" all_tests.log
```

### Interactive bash session
```bash
/remote-docker-exec src/neural/backends/sycl bash
```

## What This Skill Does

1. **Directory Check** - Verifies the local directory exists
2. **SCP Transfer** - Copies the directory to root@10.112.229.160:/home/tianfeng/test/
3. **Container Check** - Verifies cuda12.9-test container is running (starts if stopped)
4. **Execution** - Runs the command in /workspace/test/ inside the container
5. **Output Capture** - Collects stdout/stderr to the specified output file

## Remote Execution Flow

```
Local Directory → SCP → Remote Host → Docker Container → Execute → Output File
```

## Troubleshooting

### SSH Connection Issues
Ensure SSH key is set up:
```bash
ssh-copy-id root@10.112.229.160
```

### Container Not Running
The script will try to start a stopped container automatically. If the container doesn't exist:
```bash
ssh root@10.112.229.160 "docker ps -a"
```

### Path Mapping
Remember the path mapping:
- Host path: /home/tianfeng/test/<directory>
- Container path: /workspace/test/<directory>

## Alternative: Python Script

For advanced features (output parsing, JSON results, timeout):

```bash
python3 remote_docker_exec.py src/neural/backends/sycl "make" \
    --output build.log \
    --parse-output \
    --json-result result.json
```

## Implementation

This skill invokes: `./remote_docker_exec.sh $ARGUMENTS`
