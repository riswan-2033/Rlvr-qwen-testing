# Docker on Kaggle — Setup, Run, Connect & Send Code Prompts

This guide explains, end to end, how the Docker sandbox in `kaggle/` works and
how to actually get a Docker daemon running so you can use it inside Kaggle.

The sandbox is **Docker-only**: every generated code sample is executed in a
throwaway `python:3.11-slim` container. You do NOT need Docker on a phone/laptop
for this to work — you need **one reachable Docker daemon** (your own VM, a
Colab session, a laptop, or Docker-in-Docker inside the Kaggle kernel).

---

## 1. Why Docker is needed

The verifier must run *untrusted solved code* in isolation. The pipeline does:

```
prompts (from config dataset)
   │
   ▼
policy model (2x T4, device_map="auto")  ──►  generates code
   │
   ▼
Docker sandbox (python:3.11-slim container)  ──►  runs code + tests
   │                                             reward = passed/total
   ▼
GPRO trainer updates the model (clipped surrogate + KL)
```

**Terminology you will hit:**
- **Prompt** = the problem text sent to the LLM (`prompt` field, `src/dataset.py`).
- **Code** = the LLM's answer (extracted ```python block).
- **Tests** = assertion strings from the dataset.
- **Container** = the fresh Docker sandbox that runs `code + tests`.
- **Verifier** = the Docker harness that decides PASS/FAIL and produces reward.

---

## 2. How code/prompts actually flow into the container

When the trainer calls `evaluate_in_isolation(...)` (`src/sandbox.py`), it does this:

1. Builds one Python program string:
   ```
   {generated code}

   {setup_code}

   tests = ["assert solution(...) == ...", ...]
   results = []
   for t in tests:
       try: exec(t, globals()); results.append(True)
       except: results.append(False)
   print("__RLVR_VERDICT__" + json.dumps({"passed": ..., "total": ...}))
   ```
2. **Base64-encodes** that whole program so no shell quoting can break it.
3. Runs a new container:
   ```bash
   docker run --rm --network-disabled --read-only --mem-limit=512m --cpus=0.5 \
     python:3.11-slim python -c \
     "import base64; exec(base64.b64decode('<b64 payload>').decode('utf-8'))"
   ```
4. Reads the `__RLVR_VERDICT__ {...}` line → `passed/total` → **reward**.
5. Returns `{status, reward, passed, total, error, output}` to the trainer.

Nothing about prompts or code touches the disk locally on Kaggle — it all lives
in the base64 payload that is streamed into each container.

---

## 3. Prerequisites

Inside the Kaggle notebook (or the CLI), install the Python Docker SDK:

```bash
pip install -r requirements.txt      # includes docker>=6.0
```

The code connects to whatever daemon `DOCKER_HOST` points to. There are three
ways to get a daemon. Pick ONE (recommended: **Option A**).

---

## 4. Option A — Remote Docker host (RECOMMENDED, most reliable)

Run the Kaggle code in the Kaggle kernel, but execute the containers on a
separate machine where Docker already works (your laptop, a VPS, a cloud VM).

### A.1 Start Docker on the remote machine

Local Linux/macOS/Windows-with-DockerDesktop:

```bash
# Linux, with systemd:
sudo systemctl enable --now docker
# or, no systemd:
sudo dockerd &

docker info            # sanity check -> server version shown
docker pull python:3.11-slim
```

### A.2 Expose the daemon over TCP (with TLS) and connect

On the remote **Docker host** (`/etc/docker/daemon.json`):

```json
{
  "tls": true,
  "tlscacert": "/etc/docker/ca.pem",
  "tlscert": "/etc/docker/server-cert.pem",
  "tlskey": "/etc/docker/server-key.pem",
  "hosts": ["tcp://0.0.0.0:2376", "unix:///var/run/docker.sock"]
}
```

Restart: `sudo systemctl restart docker` (or `sudo killall dockerd && sudo dockerd &`).

On **Kaggle**, set the env vars and test:

```python
import os
os.environ["DOCKER_HOST"] = "tcp://<REMOTE_IP>:2376"
os.environ["DOCKER_TLS_VERIFY"] = "1"
os.environ["DOCKER_CERT_PATH"] = "."   # dir containing ca.pem, cert.pem, key.pem

from src.sandbox import get_client
print(get_client().version())          # must print the daemon version
```

> **No-TLS shortcut (dev only):** run `dockerd -H tcp://0.0.0.0:2375` on the
> remote and just set `DOCKER_HOST="tcp://<IP>:2375"` on Kaggle. Fine for testing,
> never for untrusted multi-tenant code — TLS is strongly recommended.

---

## 5. Option B — Secure shell (SSH) to a Docker host (no TLS setup)

Use the Docker CLI over SSH; the SDK follows `DOCKER_HOST` only, so we create a
context and point the SDK at it via a small shim.

On Kaggle:

```python
# 1. make an ssh docker context
!docker context create --docker "host=ssh://user@REMOTE_IP" kaggleremote

# 2. point your code at it
import os
os.environ["DOCKER_CONTEXT"] = "kaggleremote"
# docker.from_env() now tunnels over ssh (ssh key must be authorized & agent running)
```

If you'd rather avoid contexts, export directly:

```bash
export DOCKER_HOST="ssh://user@REMOTE_IP"
```

The `docker` SDK supports `ssh://` hosts **if** `ssh`/`docker-ssh` credential
helpers are present. `sshpass` + `ssh -o` options may be needed for automation.

---

## 6. Option C — Docker-in-Docker inside the Kaggle kernel

Kaggle kernels are themselves containers, so a nested daemon only works when the
kernel grants the needed privileges (root perms + `--privileged`-like caps).
Try it as a fallback:

```python
!apt-get update && apt-get install -y docker.io

# start dockerd in the background (needs root; may fail on locked-down kernels)
!nohup dockerd --host=unix:///var/run/docker.sock --storage-driver=vfs \
    --iptables=false --bridge=none > /tmp/dockerd.log 2>&1 &
import time, docker
for _ in range(15):
    try:
        c = docker.from_env(); c.version(); print("DinD OK"); break
    except Exception:
        time.sleep(3)
```

If `dockerd` won't start (permission denied on cgroups/mounts), **use Option A**.

---

## 7. Verify the connection before training

`get_client()` (`src/sandbox.py`) already validates the daemon and raises a clear
error if unreachable. Run the built-in self-test:

```python
import sys, os
sys.path.insert(0, "/kaggle/working/kaggle")   # wherever you cloned the repo

from main import run
# ---- OR just test the sandbox ----
from src.sandbox import get_client, evaluate_in_isolation
get_client()
print(evaluate_in_isolation(
    code="def solution(a,b):\n    return a+b\n",
    tests=["assert solution(1,2)==3", "assert solution(2,3)==5"],
))
# expected: {'status': 'PASS', 'reward': 1.0, 'passed': 2, 'total': 2, ...}
```

You must see the verdict, otherwise the trainer will fail at the first rollout.

---

## 8. Config knobs that affect the sandbox

Set these in **`../config/config.yaml`** (the main config the `kaggle/` code loads):

```yaml
sandbox_image: "python:3.11-slim"   # base image for every execution
execution_timeout: 5.0              # seconds before container is killed
memory_limit_mb: 512                # container memory cap
cpu_limit: "0.5"                    # 50% of one CPU core
network_disabled: true              # no internet from within the container
read_only_fs: true                  # read-only root filesystem inside container
```

---

## 9. Full run on Kaggle (end to end)

```bash
# inside a Kaggle notebook (shell cells)
!git clone https://github.com/riswan-2033/Rlvr-qwen-testing.git
!cd Rlvr-qwen-testing && pip install -q -r kaggle/requirements.txt

# Option A recommended: point at your remote daemon
import os
os.environ["DOCKER_HOST"] = "tcp://<REMOTE_IP>:2376"
os.environ["DOCKER_TLS_VERIFY"] = "1"

!cd Rlvr-qwen-testing/kaggle && python main.py --epochs 10 --samples 50
```

Or in Python cells:

```python
import os, sys
os.chdir("/kaggle/working/Rlvr-qwen-testing/kaggle")
sys.path.insert(0, ".")
os.environ["DOCKER_HOST"] = "tcp://<REMOTE_IP>:2376"
os.environ["DOCKER_TLS_VERIFY"] = "1"
from main import run
run(epochs=10, samples=50)

# or in a browser once done (Kaggle cell):
!mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0
```

## 9b. What MLflow shows after completion

The pipeline logs **everything** per run (overview + examples in README):

- **Params**: the full config from `config/config.yaml`.
- **Rollout artifacts** (`rollouts/step_N/sample_N.json`): input prompt, the raw
  generated text, extracted code, sandbox status/error, verifier reward.
- **Metrics**: `gpro_loss`, `mean_reward`, `pass_rate`, `sample_N_reward`, plus
  system metrics `system_gpu_util_pct`, `system_gpu_mem_gb`, `system_cpu_pct`,
  `system_ram_gb` — every training step.
- **Checkpoints**: `checkpoints/epoch_N.ckpt` artifacts.
- **History**: `training/training_history.json` + final loss/reward.

Backend defaults to `sqlite:///mlflow.db`; override with `MLFLOW_TRACKING_URI`.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Docker daemon is REQUIRED ... could not be reached` | `DOCKER_HOST` is wrong/unset, daemon not listening, or TLS mismatch. Check `docker info` on the host; bring it up with `dockerd`. |
| TLS errors | Copy `ca.pem`/`cert.pem`/`key.pem` where `DOCKER_CERT_PATH` points; set `DOCKER_TLS_VERIFY=1`. |
| `docker.errors.DockerException: ... ssh` | Use Option A (TCP+TLS) instead; SSH tunneling needs the right credentials. |
| `dockerd` won't start on Kaggle | Kaggle kernel lacks nested privileges → switch to Option A remote host. |
| `python:3.11-slim` image not found | `docker pull python:3.11-slim` on the Docker host first. |
| Slow: one container per sample | Expected (container spawn overhead). Lower `dataset_sample_limit`, raise `gpro_num_samples_per_prompt` only if GPU-bound, keep `execution_timeout` small. |
| Tests need numpy/scipy | Build once: `docker run python:3.11-slim python -m pip install numpy && docker commit <cid> python:3.11-slim` then set `sandbox_image: python:3.11-slim`. |
| `permission denied` on /var/run/docker.sock | Run as root on Kaggle or use TCP host from Option A. |