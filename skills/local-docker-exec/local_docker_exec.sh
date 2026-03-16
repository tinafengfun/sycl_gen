#!/bin/bash
#
# Local Docker Execution Script
# Usage: ./local_docker_exec.sh <local_directory> [command] [output_file]
#
# Example:
#   ./local_docker_exec.sh ./my_project "bash build.sh" build.log
#   ./local_docker_exec.sh ./test_code "python3 test.py" test_output.log
#

set -e

# Configuration
LOCAL_BASE_DIR="/home/intel/tianfeng/test"
DOCKER_CONTAINER="lsv-container"
DOCKER_WORKSPACE="/intel/tianfeng/test"
HOST_BASE_DIR="/home/intel/tianfeng"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions - output to stderr to avoid capturing in command substitution
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Usage information
usage() {
    cat << EOF
Usage: $0 <local_directory> [command] [output_file]

Arguments:
    local_directory    Local directory to copy and execute in
    command            Command to run inside docker (default: bash)
    output_file        File to save output (default: auto-generated with timestamp)

Examples:
    $0 ./my_project "bash build.sh" build.log
    $0 ./tests "./run_tests.py" test_results.log
    $0 ./code "make -j4" compile.log

Configuration:
    Docker Container: $DOCKER_CONTAINER
    Host Base Dir:    $HOST_BASE_DIR
    Docker Workspace: $DOCKER_WORKSPACE
    Local Test Dir:   $LOCAL_BASE_DIR
EOF
    exit 1
}

# Check if local directory exists
check_local_directory() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        log_error "Directory does not exist: $dir"
        exit 1
    fi
    log_info "Local directory verified: $dir"
}

# Get directory name
get_dir_name() {
    local dir="$1"
    basename "$dir"
}

# Copy directory to test location
copy_to_test_dir() {
    local local_dir="$1"
    local dir_name=$(get_dir_name "$local_dir")
    local target="${LOCAL_BASE_DIR}/${dir_name}"

    log_info "Creating test directory: $LOCAL_BASE_DIR"
    mkdir -p "$LOCAL_BASE_DIR"

    log_info "Copying directory: $local_dir -> $target"

    # Use rsync if available (faster, incremental), otherwise cp -r
    # Output redirected to stderr to avoid polluting return value
    if command -v rsync &> /dev/null; then
        rsync -avz --delete "$local_dir/" "$target/" >&2
    else
        rm -rf "$target" >&2
        cp -r "$local_dir" "$target" >&2
    fi

    if [[ $? -eq 0 ]]; then
        log_info "Directory copied successfully"
    else
        log_error "Failed to copy directory"
        exit 1
    fi

    # Only echo the directory name to stdout (for command substitution)
    echo "$dir_name"
}

# Check if docker container is running
check_docker_container() {
    log_info "Checking if docker container '$DOCKER_CONTAINER' is running..."

    local container_status
    container_status=$(docker ps --filter name="$DOCKER_CONTAINER" --filter status=running --format '{{.Names}}' 2>/dev/null)

    if [[ "$container_status" != "$DOCKER_CONTAINER" ]]; then
        log_error "Docker container '$DOCKER_CONTAINER' is not running"
        log_info "Available running containers:"
        docker ps --format '  - {{.Names}} ({{.Status}})'

        # Try to check if container exists but is stopped
        local stopped_container
        stopped_container=$(docker ps -a --filter name="$DOCKER_CONTAINER" --format '{{.Names}}' 2>/dev/null)

        if [[ "$stopped_container" == "$DOCKER_CONTAINER" ]]; then
            log_warn "Container exists but is stopped. Starting it..."
            docker start "$DOCKER_CONTAINER"
            sleep 2

            # Check again
            container_status=$(docker ps --filter name="$DOCKER_CONTAINER" --filter status=running --format '{{.Names}}' 2>/dev/null)
            if [[ "$container_status" != "$DOCKER_CONTAINER" ]]; then
                log_error "Failed to start container '$DOCKER_CONTAINER'"
                exit 1
            fi
            log_info "Container started successfully"
        else
            log_error "Container '$DOCKER_CONTAINER' does not exist"
            exit 1
        fi
    else
        log_info "Docker container '$DOCKER_CONTAINER' is running"
    fi
}

# Execute command in docker container
execute_in_docker() {
    local dir_name="$1"
    local command="${2:-bash}"
    local output_file="$3"
    local docker_target="${DOCKER_WORKSPACE}/${dir_name}"

    log_info "Executing command in docker: $command"
    log_info "Working directory: $docker_target"

    # Create a script to execute in container for better error handling
    local exec_script='#!/bin/bash
set -e
cd '"$docker_target"'
if [[ ! -d "'"$docker_target"'" ]]; then
    echo "ERROR: Directory '"$docker_target"' not found in container" >&2
    exit 1
fi

echo "=== Local Docker Execution ==="
echo "Working directory: $(pwd)"
echo "Command: '"$command"'"
echo "Timestamp: $(date)"
echo "================================"
echo ""

# Execute the command
'"$command"'
exit_code=$?

echo ""
echo "=== Execution Complete ==="
echo "Exit code: $exit_code"
echo "Timestamp: $(date)"
exit $exit_code
'

    # Save script and copy to container
    local local_script="${LOCAL_BASE_DIR}/.exec_script_$$.sh"
    local container_script="/tmp/.exec_script_$$.sh"
    echo "$exec_script" > "$local_script"

    # Copy script to container
    docker cp "$local_script" "$DOCKER_CONTAINER:$container_script"

    log_info "Starting execution..."
    echo "========================================"

    # Execute in docker and capture output
    if [[ -n "$output_file" ]]; then
        docker exec -i "$DOCKER_CONTAINER" bash "$container_script" > "$output_file" 2>&1
        local exit_code=$?
        echo "========================================"

        # Cleanup scripts
        rm -f "$local_script" 2>/dev/null || true
        docker exec "$DOCKER_CONTAINER" rm -f "$container_script" 2>/dev/null || true

        if [[ $exit_code -eq 0 ]]; then
            log_info "Execution completed successfully"
            log_info "Output saved to: $output_file"
        else
            log_error "Execution failed with exit code: $exit_code"
            log_info "Check output file for details: $output_file"
        fi

        # Show output summary
        local output_lines
        output_lines=$(wc -l < "$output_file" 2>/dev/null || echo "0")
        log_info "Output file contains $output_lines lines"

        return $exit_code
    else
        # Stream output to console
        docker exec -i "$DOCKER_CONTAINER" bash "$container_script"
        local exit_code=$?

        # Cleanup scripts
        rm -f "$local_script" 2>/dev/null || true
        docker exec "$DOCKER_CONTAINER" rm -f "$container_script" 2>/dev/null || true

        return $exit_code
    fi
}

# Main function
main() {
    # Validate arguments
    if [[ $# -lt 1 ]]; then
        usage
    fi

    local local_dir="$1"
    local command="${2:-bash}"
    local output_file="${3:-local_exec_$(date +%Y%m%d_%H%M%S).log}"

    # Handle relative paths
    if [[ ! "$local_dir" = /* ]]; then
        local_dir="$(pwd)/$local_dir"
    fi

    log_info "=== Local Docker Execution ==="
    log_info "Local directory: $local_dir"
    log_info "Docker container: $DOCKER_CONTAINER"
    log_info "Command: $command"
    log_info "Output file: $output_file"
    echo ""

    # Step 1: Check local directory
    check_local_directory "$local_dir"

    # Step 2: Copy to test directory
    local dir_name=$(copy_to_test_dir "$local_dir")

    # Step 3: Check docker container
    check_docker_container

    # Step 4: Execute command
    echo ""
    execute_in_docker "$dir_name" "$command" "$output_file"
    local exit_code=$?

    echo ""
    log_info "=== Execution Summary ==="
    log_info "Directory: $dir_name"
    log_info "Command: $command"
    if [[ -n "$output_file" ]]; then
        log_info "Output: $output_file"
    fi

    if [[ $exit_code -eq 0 ]]; then
        log_info "Status: SUCCESS"
    else
        log_error "Status: FAILED (exit code $exit_code)"
    fi

    return $exit_code
}

# Run main function
main "$@"
