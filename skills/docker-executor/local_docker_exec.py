#!/usr/bin/env python3
"""
Local Docker Execution Script

Usage:
    python3 local_docker_exec.py <local_directory> [command] [options]

Examples:
    python3 local_docker_exec.py ./my_project "bash build.sh"
    python3 local_docker_exec.py ./tests "pytest" --output test_results.log
    python3 local_docker_exec.py ./code "make" --parse-output --json-result result.json
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


class LocalDockerExecutor:
    """Execute commands in local Docker container with output capture and analysis."""

    # Configuration
    LOCAL_BASE_DIR = "/home/intel/tianfeng/test"
    DOCKER_CONTAINER = "lsv-container"
    DOCKER_WORKSPACE = "/intel/tianfeng/test"
    HOST_BASE_DIR = "/home/intel/tianfeng"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.execution_log = []

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color_map = {
            "INFO": "\033[0;32m",
            "WARN": "\033[1;33m",
            "ERROR": "\033[0;31m",
            "DEBUG": "\033[0;36m",
        }
        reset = "\033[0m"
        color = color_map.get(level, "")

        log_entry = f"[{timestamp}] [{level}] {message}"
        if level == "DEBUG" and not self.verbose:
            self.execution_log.append(log_entry)
            return

        print(f"{color}[{level}]{reset} {message}")
        self.execution_log.append(log_entry)

    def run_command(
        self, command: list, capture_output: bool = True, timeout: Optional[int] = None
    ) -> Tuple[int, str, str]:
        """Run a local command."""
        if self.verbose:
            self.log(f"Command: {' '.join(command)}", "DEBUG")

        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", str(e)

    def copy_directory(self, src: Path, dst: Path) -> bool:
        """Copy directory using rsync or shutil."""
        self.log(f"Copying {src} -> {dst}")

        # Try rsync first
        rsync_available = shutil.which("rsync") is not None

        if rsync_available:
            cmd = ["rsync", "-avz", "--delete", f"{src}/", f"{dst}/"]
        else:
            # Fallback to shutil
            if dst.exists():
                shutil.rmtree(dst)
            cmd = ["cp", "-r", str(src), str(dst)]

        rc, stdout, stderr = self.run_command(cmd)
        if rc != 0:
            self.log(f"Copy failed: {stderr}", "ERROR")
            return False

        return True

    def check_local_directory(self, local_dir: str) -> Path:
        """Validate local directory exists."""
        path = Path(local_dir).resolve()
        if not path.exists():
            raise ValueError(f"Directory does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        self.log(f"Local directory verified: {path}")
        return path

    def copy_to_test_dir(self, local_dir: Path) -> str:
        """Copy directory to test location."""
        dir_name = local_dir.name
        target = Path(self.LOCAL_BASE_DIR) / dir_name

        # Create base directory
        Path(self.LOCAL_BASE_DIR).mkdir(parents=True, exist_ok=True)

        if not self.copy_directory(local_dir, target):
            raise RuntimeError("Failed to copy directory")

        self.log("Directory copied successfully")
        return dir_name

    def check_docker_container(self) -> bool:
        """Check if docker container is running, start if stopped."""
        self.log(f"Checking docker container '{self.DOCKER_CONTAINER}'...")

        # Check if running
        cmd = [
            "docker", "ps",
            "--filter", f"name={self.DOCKER_CONTAINER}",
            "--filter", "status=running",
            "--format", "{{.Names}}"
        ]
        rc, stdout, _ = self.run_command(cmd)

        if stdout.strip() == self.DOCKER_CONTAINER:
            self.log(f"Container '{self.DOCKER_CONTAINER}' is running")
            return True

        # Check if exists but stopped
        cmd = [
            "docker", "ps", "-a",
            "--filter", f"name={self.DOCKER_CONTAINER}",
            "--format", "{{.Names}}"
        ]
        rc, stdout, _ = self.run_command(cmd)

        if stdout.strip() == self.DOCKER_CONTAINER:
            self.log("Container exists but is stopped. Starting...", "WARN")
            self.run_command(["docker", "start", self.DOCKER_CONTAINER])

            # Verify it started
            cmd = [
                "docker", "ps",
                "--filter", f"name={self.DOCKER_CONTAINER}",
                "--filter", "status=running",
                "--format", "{{.Names}}"
            ]
            rc, stdout, _ = self.run_command(cmd)
            if stdout.strip() == self.DOCKER_CONTAINER:
                self.log("Container started successfully")
                return True
            else:
                raise RuntimeError("Failed to start container")

        raise RuntimeError(f"Container '{self.DOCKER_CONTAINER}' does not exist")

    def execute_in_docker(
        self, dir_name: str, command: str, timeout: Optional[int] = None
    ) -> Tuple[int, str, str]:
        """Execute command in docker container."""
        docker_target = f"{self.DOCKER_WORKSPACE}/{dir_name}"

        self.log(f"Executing in docker: {command}")
        self.log(f"Working directory: {docker_target}")

        # Create execution script
        script = f"""#!/bin/bash
set -e
cd {docker_target}
if [[ ! -d '{docker_target}' ]]; then
    echo "ERROR: Directory {docker_target} not found in container" >&2
    exit 1
fi

echo "=== Local Docker Execution ==="
echo "Working directory: $(pwd)"
echo "Command: {command}"
echo "Timestamp: $(date)"
echo "================================"
echo ""

{command}
exit_code=$?

echo ""
echo "=== Execution Complete ==="
echo "Exit code: $exit_code"
echo "Timestamp: $(date)"
exit $exit_code
"""

        # Save script and copy to container
        local_script = Path(self.LOCAL_BASE_DIR) / f".exec_script_{os.getpid()}.sh"
        container_script = f"/tmp/.exec_script_{os.getpid()}.sh"
        local_script.write_text(script)

        # Copy to container
        self.run_command(["docker", "cp", str(local_script), f"{self.DOCKER_CONTAINER}:{container_script}"])

        self.log("Starting execution...")
        print("=" * 60)

        # Execute in docker
        docker_cmd = ["docker", "exec", "-i", self.DOCKER_CONTAINER, "bash", container_script]
        rc, stdout, stderr = self.run_command(docker_cmd, capture_output=True, timeout=timeout)

        print("=" * 60)

        # Cleanup
        local_script.unlink(missing_ok=True)
        self.run_command(["docker", "exec", self.DOCKER_CONTAINER, "rm", "-f", container_script])

        return rc, stdout, stderr

    def parse_output(self, stdout: str, stderr: str) -> dict:
        """Parse execution output for analysis."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "stdout_lines": len(stdout.splitlines()) if stdout else 0,
            "stderr_lines": len(stderr.splitlines()) if stderr else 0,
            "errors": [],
            "warnings": [],
            "summary": {},
        }

        combined_output = f"{stdout}\n{stderr}"

        # Look for common error patterns
        error_patterns = [
            r"error:\s*(.+)",
            r"ERROR:\s*(.+)",
            r"FAILED\s*(.+)",
            r"CMake Error.*",
            r"make: \*\*\*.*",
        ]

        warning_patterns = [
            r"warning:\s*(.+)",
            r"WARNING:\s*(.+)",
        ]

        for pattern in error_patterns:
            matches = re.findall(pattern, combined_output, re.IGNORECASE)
            result["errors"].extend(matches)

        for pattern in warning_patterns:
            matches = re.findall(pattern, combined_output, re.IGNORECASE)
            result["warnings"].extend(matches)

        # Try to extract build statistics
        if "make" in combined_output.lower():
            target_match = re.search(r"Built target\s+(\w+)", combined_output)
            if target_match:
                result["summary"]["built_targets"] = target_match.group(1)

        # Test stats
        test_passed = len(re.findall(r"\[\s*PASSED\s*\]|passed|✓", combined_output, re.IGNORECASE))
        test_failed = len(re.findall(r"\[\s*FAILED\s*\]|failed|✗", combined_output, re.IGNORECASE))
        if test_passed or test_failed:
            result["summary"]["tests_passed"] = test_passed
            result["summary"]["tests_failed"] = test_failed

        return result

    def run(
        self,
        local_dir: str,
        command: str,
        output_file: Optional[str] = None,
        parse_output: bool = False,
        json_result: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> int:
        """Main execution flow."""
        try:
            # Step 1: Check local directory
            path = self.check_local_directory(local_dir)

            # Step 2: Copy to test directory
            dir_name = self.copy_to_test_dir(path)

            # Step 3: Check docker
            self.check_docker_container()

            # Step 4: Execute
            print()
            rc, stdout, stderr = self.execute_in_docker(dir_name, command, timeout)

            # Save output if requested
            if output_file:
                with open(output_file, "w") as f:
                    f.write("=== STDOUT ===\n")
                    f.write(stdout)
                    f.write("\n=== STDERR ===\n")
                    f.write(stderr)
                self.log(f"Output saved to: {output_file}")

            # Parse output if requested
            analysis = None
            if parse_output or json_result:
                analysis = self.parse_output(stdout, stderr)
                self.log(f"Errors found: {len(analysis['errors'])}")
                self.log(f"Warnings found: {len(analysis['warnings'])}")

            # Save JSON result if requested
            if json_result:
                result = {
                    "success": rc == 0,
                    "exit_code": rc,
                    "directory": str(path),
                    "command": command,
                    "analysis": analysis,
                    "stdout_preview": stdout[:2000] if stdout else "",
                    "stderr_preview": stderr[:2000] if stderr else "",
                }
                with open(json_result, "w") as f:
                    json.dump(result, f, indent=2)
                self.log(f"JSON result saved to: {json_result}")

            # Print summary
            print()
            self.log("=== Execution Summary ===")
            self.log(f"Directory: {dir_name}")
            self.log(f"Command: {command}")
            self.log(f"Exit Code: {rc}")

            if rc == 0:
                self.log("Status: SUCCESS")
            else:
                self.log("Status: FAILED", "ERROR")

            return rc

        except Exception as e:
            self.log(f"Execution failed: {e}", "ERROR")
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Execute commands in local Docker container (lsv-container)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s ./my_project "bash build.sh"
    %(prog)s ./tests "pytest -v" --output test.log --parse-output
    %(prog)s ./code "make -j4" --json-result build_result.json
        """,
    )

    parser.add_argument("directory", help="Local directory to copy and execute")
    parser.add_argument("command", nargs="?", default="bash", help="Command to run (default: bash)")
    parser.add_argument("-o", "--output", help="Save output to file")
    parser.add_argument("-p", "--parse-output", action="store_true", help="Parse output for errors/warnings")
    parser.add_argument("-j", "--json-result", help="Save structured result to JSON file")
    parser.add_argument("-t", "--timeout", type=int, help="Timeout in seconds")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    executor = LocalDockerExecutor(verbose=args.verbose)
    return executor.run(
        local_dir=args.directory,
        command=args.command,
        output_file=args.output,
        parse_output=args.parse_output,
        json_result=args.json_result,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
