# =============================================================================
# Docker sandbox verifier for the RLVR code-verifier (Kaggle edition)
#
# STRICT DOCKER-ONLY MODE:
#   Every generated code sample runs inside its own throwaway container.
#   No multiprocessing / CPU-exec fallback is used.
#
# Reward model (verifier):
#   For MBPP: each test in `tests` is executed; reward = fraction passing.
#   For HumanEval: the bundled `test` string either passes (1.0) or fails (0.0).
#
# Result dict:
#   { status, reward, passed, total, error, output }
#   status: "PASS" | "FAIL" | "TIMEOUT" | "CRASH"
# =============================================================================
import base64
import json
import docker
from typing import Dict, Any, List, Optional


def get_client() -> docker.DockerClient:
    """Return a verified Docker client or raise if the daemon is unavailable.

    Connection honours the standard env vars so you can point at a remote or
    Docker-in-Docker daemon without code changes:
      DOCKER_HOST         e.g. tcp://<ip>:2376   (remote) or unix socket
      DOCKER_TLS_VERIFY=1, DOCKER_CERT_PATH     (TLS client certs)
      DOCKER_CONTEXT      any docker context name
    """
    try:
        client = docker.from_env()
        client.version()
        return client
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Docker daemon is REQUIRED for this sandbox but could not be reached: "
            f"{e}\n"
            "Check DOCKER_HOST / DOCKER_CONTEXT and see kaggle/DOCKER_SETUP_KAGGLE.md."
        )


def _build_script(
    code: str,
    tests: List[str],
    setup_code: str = "",
) -> str:
    """Compose the program executed inside the container.

    The generated `code` is embedded first, then setup code, then a small
    harness that runs every test independently and prints a JSON verdict line.
    """
    tests_json = json.dumps(tests)
    harness = f"""
import json, traceback

tests = {tests_json}
results = []
for idx, t in enumerate(tests):
    try:
        exec(t, globals())
        results.append(True)
    except Exception as e:  # noqa: BLE001
        results.append(False)

passed = sum(1 for r in results if r)
print("__RLVR_VERDICT__" + json.dumps({{"passed": passed, "total": len(tests)}}))
"""
    return f"{code}\n\n{setup_code}\n{harness}"


def _run_container(
    client: docker.DockerClient,
    script: str,
    image: str,
    timeout: float,
    memory_mb: int,
    cpu_limit: str,
    network_disabled: bool,
    read_only_fs: bool,
) -> bytes:
    """Run script in a fresh container; returns stdout bytes."""
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    command = [
        "python",
        "-c",
        "import base64; exec(base64.b64decode('" + encoded + "').decode('utf-8'))",
    ]
    return client.containers.run(
        image=image,
        command=command,
        mem_limit=f"{memory_mb}m",
        cpus=cpu_limit,
        network_disabled=network_disabled,
        read_only=read_only_fs,
        pids_limit=128,
        auto_remove=True,
        detach=False,
        stdout=True,
        stderr=True,
        timeout=timeout,
        tty=False,
    )


def evaluate_in_isolation(
    code: str,
    tests: List[str],
    setup_code: str = "",
    image: str = "python:3.11-slim",
    timeout: float = 5.0,
    memory_mb: int = 512,
    cpu_limit: str = "0.5",
    network_disabled: bool = True,
    read_only_fs: bool = True,
    client: Optional[docker.DockerClient] = None,
) -> Dict[str, Any]:
    """Execute generated code against its tests inside Docker and return a verdict."""
    client = client or get_client()
    script = _build_script(code, tests, setup_code)

    try:
        output = _run_container(
            client, script, image, timeout, memory_mb, cpu_limit,
            network_disabled, read_only_fs,
        )
        text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)

        verdict = None
        for line in text.splitlines():
            marker = "__RLVR_VERDICT__"
            if marker in line:
                verdict = json.loads(line.split(marker, 1)[1])
                break

        if verdict is None:
            # No harness output -> crash/exit without printing verdict.
            return {
                "status": "CRASH",
                "reward": 0.0,
                "passed": 0,
                "total": len(tests),
                "error": "No verdict produced by container",
                "output": text[:2000],
            }

        passed, total = verdict["passed"], verdict["total"]
        reward = (passed / total) if total > 0 else 1.0
        status = "PASS" if reward == 1.0 else "FAIL"
        return {
            "status": status,
            "reward": reward,
            "passed": passed,
            "total": total,
            "error": None,
            "output": text[:2000],
        }

    except docker.errors.Timeout:
        return {
            "status": "TIMEOUT",
            "reward": 0.0,
            "passed": 0,
            "total": len(tests),
            "error": "Execution exceeded Docker timeout",
            "output": None,
        }
    except docker.errors.ContainerError as e:
        err = (e.stderr or str(e))
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="replace")
        return {
            "status": "FAIL",
            "reward": 0.0,
            "passed": 0,
            "total": len(tests),
            "error": str(err)[:1000],
            "output": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "status": "CRASH",
            "reward": 0.0,
            "passed": 0,
            "total": len(tests),
            "error": f"Unexpected sandbox error: {e}",
            "output": None,
        }
