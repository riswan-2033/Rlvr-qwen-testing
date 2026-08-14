# Docker Sandbox Setup for RLVR Code Verifier

This document provides end-to-end instructions for setting up Docker-based code execution sandboxing for the RLVR (Reinforcement Learning from Human Feedback with Verifier) code verifier system.

---

## 1. Overview

The current `sandbox.py` uses **process-level sandboxing** with OS resource limits (`resource.RLIMIT_CPU`, `resource.RLIMIT_AS`). This is NOT a full container isolation.

To achieve proper security isolation, we'll replace this with **Docker-based execution** where each code sample runs in its own disposable container.

---

## 2. Docker Requirements

- Docker installed and running
- User added to `docker` group (or run with `sudo`)
- Sufficient disk space for container layers
- Network access to pull Python base images

---

## 3. Directory Structure

Create a `docker` directory in the project root:

```
/home/blu-bridge044/Desktop/RISWAN-AHAMED/Implementation/rlvr-code-verifier/
├── docker/
│   ├── Dockerfile        # Container build definition
│   └── sandbox.py        # Updated sandbox implementation
├── src/
│   ├── model_runner.py
│   ├── dataset.py
│   └── tracker.py
├── config/
│   └── config.yaml
└── main.py
```

---

## 4. Dockerfile

Create `docker/Dockerfile`:

```dockerfile
# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd -m runner
USER runner

# Default command
CMD ["python", "-v"]
```

---

## 4. requirements.txt

Create `docker/requirements.txt`:

```
transformers>=4.35.0
torch>=2.0.0
datasets>=4.8.0
torch.multiprocessing
multiprocess
```

---

## 5. Updated sandbox.py (Docker-based)

Create `docker/sandbox.py`:

```python
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
        "output": output.decode('utf-8') if isinstance(output, bytes) else output,
        "error": None
    }

    return result

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
```

---

## 6. Updated main.py Integration

Update `main.py` to use the Docker sandbox:

```python
from src.sandbox import evaluate_in_isolation

# ... rest of main() stays the same ...

# The sandbox is now called via:
result = evaluate_in_isolation(
    code_string=generated_code,
    assertions=item["test_assertions"],
    timeout=config["execution_timeout"],
    memory_mb=config["memory_limit_mb"]
)
```

---

## 7. Building the Docker Image

Run once to build the custom container:

```bash
cd /home/blu-bridge044/Desktop/RISWAN-AHAMED/Implementation/rlvr-code-verifier
docker build -t rlvr-sandbox:latest -f docker/Dockerfile .
```

This creates an image `rlvr-sandbox:latest` that:
- Has Python 3.11 installed
- Includes required packages (transformers, torch, datasets)
- Runs as non-root user
- Has resource limits configured

---

## 8. How It Works - End-to-End Flow

### 8.1 Code Generation Phase
1. `main.py` loads config from `config/config.yaml`
2. `LocalLLMRunner` loads `Qwen/Qwen2.5-Coder-0.5B-Instruct` model
3. Model generates code solutions for each prompt batch

### 8.2 Sandbox Execution Phase
1. For each generated code:
   - `evaluate_in_isolation()` is called with:
     - `code_string`: The generated Python function
     - `assertions`: Test assertions from dataset
     - `timeout`: From config (default 2.0s)
     - `memory_mb`: From config (default 256MB)
   
2. Docker sandbox creates a container:
   - Runs `python -c "code_string + assertions"`
   - Memory limited to 256MB
   - CPU limited to 50% of one core
   - No network access (isolated)
   - Read-only filesystem

3. Container execution results:
   - **PASS** (reward 1.0): Code executed without errors
   - **FAIL** (reward 0.0): Exception during execution
   - **TIMEOUT** (reward 0.0): Exceeded time limit
   - **CRASH** (reward 0.0): Container crashed

### 8.3 Result Processing
1. Result dictionary contains:
   - `status`: PASS/FAIL/TIMEOUT/CRASH
   - `reward`: 1.0 or 0.0
   - `error`: Error message (if any, truncated to 1000 chars)
   - `output`: Stdout from execution (optional)

2. Result is logged to MLflow:
   ```python
   tracker.log_evaluation_transaction(
       step_idx=step_idx,
       sample_id=global_sample_id,
       prompt=item["prompt"],
       generated_code=generated_code,
       verification_result=result
   )
   ```

### 8.4 Reward Mechanism
- **Reward = 1.0**: Code passes all assertions without errors
- **Reward = 0.0**: Code fails any assertion or exceeds limits
- The reward is logged as an MLflow metric: `reward_sample_{sample_id}`

---

## 9. Running the System with Docker

### 9.1 Start Docker
```bash
# On Linux
sudo systemctl start docker
# Or add user to docker group
usermod -aG docker $USER
newgrp docker
```

### 9.2 Build the Sandbox Image
```bash
cd /home/blu-bridge044/Desktop/RISWAN-AHAMED/Implementation/rlvr-code-verifier
docker build -t rlvr-sandbox:latest -f docker/Dockerfile .
```

### 9.3 Test the Docker Sandbox
```bash
docker run --rm rlvr-sandbox:latest python -c "print('Hello from Docker')"
```

### 9.4 Run the Full Pipeline
```bash
cd /home/blu-bridge044/Desktop/RISWAN-AHAMED/Implementation/rlvr-code-verifier
python main.py
```

---

## 10. Security Considerations

| Feature | Protection Level |
|---------|-----------------|
| `read_only` filesystem | Prevents code from writing files |
| `mem_limit` | Limits memory consumption |
| `cpus` | Prevents CPU starvation |
| `network_disabled` | No outbound network from code |
| `auto_remove` | Container cleanup after exit |
| `pid_limit` | Prevents fork bombs |
| `rm` flag | Auto-remove container on exit |

### Known Limitations
- Docker socket access requires proper permissions
- Container startup overhead (~1-2 seconds per execution)
- Not all Python packages may be available in the slim image
- Changes required to `docker/requirements.txt` if using additional packages

---

## 11. Troubleshooting

### Common Issues

1. **Docker permission denied**
   - Add user to docker group: `usermod -aG docker $USER`
   - Log out and back in, or run `newgrp docker`

2. **Container fails to start**
   - Check: `docker logs <container_id>`
   - Ensure `docker/requirements.txt` has correct packages

3. **Memory limit errors**
   - Increase `memory_limit_mb` in `config/config.yaml`
   - Or reduce batch size

4. **Timeout errors**
   - Increase `execution_timeout` in `config/config.yaml`
   - Or optimize generated code

5. **Package import errors**
   - Add required packages to `docker/requirements.txt`
   - Rebuild: `docker build -t rlvr-sandbox:latest -f docker/Dockerfile .`

---

## 12. Performance Notes

- **Container startup time**: ~500ms - 2s per execution
- **Memory overhead**: ~50-100MB per container (on top of code)
- **CPU overhead**: Minimal, limits enforce CPU usage caps
- **Total pipeline time**: Each batch step includes model generation + container execution + result logging

For production use, consider:
- Container reuse/pooling to amortize startup cost
- Parallel container execution (Docker Swarm/K8s)
- Persistent volume for model caching
- Enhanced security profiles (seccomp, AppArmor)