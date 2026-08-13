import docker
import json
import time
from typing import Dict, Any

# Docker client initialization with fallback
docker_client = None
try:
    docker_client = docker.DockerClient()
    # Test connection
    docker_client.version()
    print("Docker client connected successfully")
except (docker.errors.DockerException, PermissionError, ConnectionError) as e:
    try:
        docker_client = docker.DockerClient(base_url='unix://var/run/docker.sock')
        docker_client.version()
        print("Docker client connected via explicit socket")
    except Exception:
        docker_client = None
        print("Docker unavailable - sandbox will use multiprocessing fallback")
except Exception as e:
    docker_client = None
    print(f"Docker initialization error: {e}")

# Resource limits per container
DEFAULT_MEMORY_MB = 256
DEFAULT_TIMEOUT = 2.0  # seconds


def enforce_limits(memory_mb: int):
    """Returns Docker resource constraints dictionary."""
    return {
        "memory": f"{memory_mb}m",
        "cpu_quota": 50000,  # 50% of one CPU core
        "cpu_period": 100000,
        "mem_limit": f"{memory_mb}m",
    }


def execute_in_container(code_string: str, assertions: str, timeout: float, memory_mb: int) -> Dict[str, Any]:
    """
    Execute untrusted Python code inside a Docker container.
    
    This provides full OS-level isolation compared to the previous
    multiprocessing-based sandbox.
    
    Args:
        code_string: The generated Python code to execute
        assertions: Test assertions to append
        timeout: Maximum execution time in seconds
        memory_mb: Memory limit in MB
        
    Returns:
        Dictionary with status, reward, error, and optional output
    """
    # If Docker not available, fall back to multiprocessing sandbox
    if docker_client is None:
        from .model_runner import LocalLLMRunner
        # Use basic multiprocessing execution as fallback
        return _multiprocessing_fallback(code_string, assertions, timeout, memory_mb)
    
    try:
        # Merge model solution code directly with verification assertions
        full_execution_script = f"{code_string}\n\n{assertions}"
        
        # Create container configuration for isolated execution
        container_kwargs = {
            "image": "python:3.11-slim",
            "command": ["python", "-c", full_execution_script],
            "remove": True,                  # Auto-remove container when it exits
            "auto_remove": True,
            "network_disabled": True,        # No network access from container
            "mem_limit": f"{memory_mb}m",    # Memory limit
            "cpu_quota": 50000,              # 50% of one CPU core
            "cpu_period": 100000,
            "timeout": timeout,              # Docker container timeout
            "auto_remove": True,
            "privileged": False,
            "read_only": True,               # Read-only filesystem for security
            "pids_limit": 500,               # Limit PID count to prevent fork bombs
            "user": "root",                  # Run as root to avoid permission issues inside container
        }
        
        # Run container and capture output
        try:
            container_output = docker_client.containers.run(**container_kwargs)
            
            # Decode output (may be bytes or string)
            if isinstance(container_output, bytes):
                output_text = container_output.decode('utf-8', errors='replace')
            else:
                output_text = str(container_output)
            
            return {
                "status": "PASS" if output_text.strip() else "FAIL",
                "reward": 1.0 if output_text.strip() else 0.0,
                "error": None,
                "output": output_text.strip() if output_text.strip() else None
            }
            
        except docker.errors.Timeout:
            return {
                "status": "TIMEOUT",
                "reward": 0.0,
                "error": "Execution exceeded Docker timeout",
                "output": None
            }
            
        except docker.errors.ContainerError as e:
            # Extract error details from container
            stderr = b""
            if hasattr(e, 'stderr') and e.stderr:
                stderr = e.stderr
            if isinstance(stderr, bytes):
                error_text = stderr.decode('utf-8', errors='replace')
            else:
                error_text = str(e)
            
            # Truncate error message
            error_msg = error_text[:1000] if len(error_text) > 1000 else error_text
            
            return {
                "status": "FAIL",
                "reward": 0.0,
                "error": error_msg,
                "output": None
            }
            
    except docker.errors.ImageNotFound:
        return {
            "status": "CRASH",
            "reward": 0.0,
            "error": "Docker image 'python:3.11-slim' not found. Please build the sandbox image first.",
            "output": None
        }
    except docker.errors.APIError as e:
        return {
            "status": "CRASH",
            "reward": 0.0,
            "error": f"Docker API error: {str(e)}",
            "output": None
        }
    except Exception as e:
        return {
            "status": "CRASH",
            "reward": 0.0,
            "error": f"Unexpected error: {str(e)}",
            "output": None
        }


def _multiprocessing_fallback(code_string: str, assertions: str, timeout: float, memory_mb: int) -> Dict[str, Any]:
    """
    Fallback execution using multiprocessing when Docker is unavailable.
    Based on the original sandbox.py implementation.
    """
    import multiprocessing
    import resource
    
    def execute_worker(code_payload: str, test_assertions: str, communication_pipe) -> None:
        try:
            enforce_limits(memory_mb=memory_mb)
            execution_scope: Dict[str, Any] = {}
            
            # Merge model solution code directly with verification assertions
            full_execution_script = f"{code_payload}\n\n{test_assertions}"
            
            exec(full_execution_script, {}, execution_scope)
            communication_pipe.send({"status": "PASS", "reward": 1.0, "error": None})
        except Exception as exc:
            communication_pipe.send({"status": "FAIL", "reward": 0.0, "error": str(exc)})
        except SystemExit:
            communication_pipe.send({"status": "CRASH", "reward": 0.0, "error": "SystemExit intercepted"})
    
    try:
        parent_conn, child_conn = multiprocessing.Pipe()
        worker_process = multiprocessing.Process(
            target=execute_worker, 
            args=(code_string, assertions, child_conn)
        )
        worker_process.start()
        child_conn.close()
        
        if parent_conn.poll(timeout):
            try:
                execution_outcome = parent_conn.recv()
            except EOFError:
                execution_outcome = {"status": "CRASH", "reward": 0.0, "error": "Abrupt execution termination"}
        else:
            worker_process.terminate()
            execution_outcome = {"status": "TIMEOUT", "reward": 0.0, "error": "Execution exceeded timeout"}
            
        worker_process.join()
        return execution_outcome
        
    except Exception as e:
        return {
            "status": "CRASH",
            "reward": 0.0,
            "error": f"Fallback error: {str(e)}",
            "output": None
        }


def evaluate_in_isolation(code_string: str, assertions: str, timeout: float = 2.0, memory_mb: int = 256) -> Dict[str, Any]:
    """
    Execute untrusted Python code in an isolated Docker container.
    
    This replaces the multiprocessing-based sandbox from sandbox.py.
    
    Args:
        code_string: The generated Python code to execute
        assertions: Test assertions to append
        timeout: Maximum execution time in seconds (default 2.0)
        memory_mb: Memory limit in MB (default 256)
        
    Returns:
        Dictionary with keys: status, reward, error, output
        - status: "PASS", "FAIL", "TIMEOUT", or "CRASH"
        - reward: 1.0 for PASS, 0.0 otherwise
        - error: Error message (truncated to 1000 chars) or None
        - output: Captured stdout (or None if timeout/error)
    """
    return execute_in_container(code_string, assertions, timeout, memory_mb)