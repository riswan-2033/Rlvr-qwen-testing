import docker
import json
import time
from typing import Dict, Any, Optional

# Docker client - connects to local Docker daemon
docker_client = docker.DockerClient(base_url='unix://var/run/docker.sock')

# Resource limits per container
DEFAULT_MEMORY_MB = 256
DEFAULT_TIMEOUT = 2.0  # seconds


def enforce_limits(memory_mb: int, timeout: float):
    """Returns Docker resource constraints."""
    return {
        "memory": f"{memory_mb}m",
        "cpu_quota": 50000,  # 50% of one CPU core
        "cpu_period": 100000,
        "mem_limit": f"{memory_mb}m",
    }


def execute_in_container(code_string: str, assertions: str, timeout: float, memory_mb: int) -> Dict[str, Any]:
    """
    Execute untrusted Python code inside a Docker container.
    
    Args:
        code_string: The generated Python code to execute
        assertions: Test assertions to append
        timeout: Maximum execution time in seconds
        memory_mb: Memory limit in MB
        
    Returns:
        Dictionary with status, reward, and error info
    """
    try:
        # Merge code with assertions
        full_script = f"{code_string}\n\n{assertions}"
        
        # Create container configuration
        container_config = {
            "image": "python:3.11-slim",
            "command": ["python", "-c", full_script],
            "remove": True,        # Auto-remove container when it exits
            "network_disabled": True,  # No network access
            "mem_limit": f"{memory_mb}m",
            "cpus": "0.5",       # Limit to 50% of one CPU
            "timeout": timeout,  # Docker container timeout
            "auto_remove": True,
            "privileged": False,
            "read_only": True,   # Read-only filesystem
            "pids_limit": 500,   # Limit PID count
        }
        
        # Run container
        container = docker_client.containers.run(**container_config)
        
        # Capture output
        result = {
            "status": "PASS",
            "reward": 1.0,
            "output": container.decode('utf-8') if isinstance(container, bytes) else container,
            "error": None
        }
        
        return result
        
    except docker.errors.Timeout:
        return {
            "status": "TIMEOUT",
            "reward": 0.0,
            "error": "Execution exceeded Docker timeout",
            "output": None
        }
    except docker.errors.ContainerError as e:
        error_msg = str(e)
        # Extract stderr if available
        stderr = getattr(e, 'stderr', None)
        if stderr:
            error_msg = stderr.decode('utf-8') if isinstance(stderr, bytes) else stderr
        
        return {
            "status": "FAIL",
            "reward": 0.0,
            "error": error_msg[:1000] if error_msg else "Container execution error",
            "output": None
        }
    except Exception as e:
        return {
            "status": "CRASH",
            "reward": 0.0,
            "error": f"Unexpected error: {str(e)}",
            "output": None
        }


def evaluate_in_isolation(code_string: str, assertions: str, timeout: float = 2.0, memory_mb: int = 256) -> Dict[str, Any]:
    """
    Public API: Execute code in Docker container with isolation.
    
    This replaces the multiprocessing-based sandbox.
    """
    return execute_in_container(code_string, assertions, timeout, memory_mb)