# Kaggle RLVR Code Verifier

Self-contained Reinforcement Learning from Verifier (RLVR) pipeline tuned for Kaggle
(**2x T4 GPUs, Docker-only sandbox**). Focus: **coding generation checking** (MBPP /
HumanEval), not math.

```
kaggle/
├── requirements.txt
├── download_models.py   # pre-download models (optional)
├── main.py              # CLI + importable entry point
└── src/
    ├── dataset.py       # MBPP / HumanEval loaders
    ├── model_runner.py  # policy rollout generation (both GPUs)
    ├── sandbox.py       # DOCKER-ONLY verifier (reward = tests passed)
    └── gpro_trainer.py  # GPRO-style policy update using verifier rewards
```

**Config**: there is NO separate config in this folder. All settings (dataset, model,
GPRO, sandbox) live in the main project config at `../config/config.yaml`, which both
`main.py` and `download_models.py` load automatically. Override with `--config <path>`.

## How it works

1. **Generate rollouts** — policy LLM writes code for each prompt
   (`num_samples_per_prompt` samples per prompt). Model sharded across both T4s
   via `device_map="auto"`; generation runs on both GPUs.
2. **Verify in Docker** — every generated code block runs in its own throwaway
   `python:3.11-slim` container (network off, read-only FS, mem/CPU limits).
3. **Reward** — verifier score `= passed_tests / total_tests` in `[0, 1]`.
4. **Update the model** — GPRO clipped surrogate loss + KL against a frozen
   reference model, using group-wise (per-prompt) advantages.
5. **Log everything to MLflow** — inputs, generated text, outputs, metrics,
   system metrics, and checkpoints (see next section).

## MLflow logging — what you get in the UI

Run `mlflow ui` after training and open the printed URL. Every run contains:

| What | Where in MLflow |
|---|---|
| Full config (dataset, model, GPRO, sandbox) | **Params** + `config/config.json` artifact |
| Every prompt sent to the LLM | `rollouts/step_N/sample_N.json` → `input_prompt` |
| The exact raw text the model generated | `rollouts/step_N/sample_N.json` → `raw_generated_text` |
| Extracted code block | `rollouts/step_N/sample_N.json` → `extracted_code` |
| Sandbox verdict + error | `rollouts/step_N/sample_N.json` → `sandbox_status`, `sandbox_error` |
| Per-sample reward | Metric `sample_N_reward` |
| Step-level loss, mean reward, pass rate | Metrics `gpro_loss`, `mean_reward`, `pass_rate` |
| System metrics each step (GPU util, GPU mem, CPU %, RAM) | `system_gpu_util_pct`, `system_gpu_mem_gb`, `system_cpu_pct`, `system_ram_gb` |
| Epoch summaries | `epoch_avg_loss`, `epoch_avg_reward` |
| **Model checkpoints** | `checkpoints/epoch_N.ckpt` artifact |
| Training history JSON | `training/training_history.json` |

Backend defaults to `sqlite:///mlflow.db` in the `kaggle/` dir (overridable via
`MLFLOW_TRACKING_URI` or `config.yaml: mlflow_tracking_uri`).

## Quick start (CLI)

```bash
cd kaggle

# 1 deps
pip install -r requirements.txt

# 2 docker image (required - sandbox is Docker-only)
docker pull python:3.11-slim
# if your Kaggle env needs to *start* the daemon: docker daemon (or sudo dockerd &)

# 3 (optional) pre-download models so training doesn't stall on network
python download_models.py

# 4 run training (uses ../config/config.yaml)
python main.py --epochs 10 --samples 50

# 5 inspect results
mlflow ui --backend-store-uri sqlite:///mlflow.db
#   -> open http://localhost:5000 in a browser (or the URL mlflow prints)
```

## Quick start (Kaggle notebook)

```python
import os
os.chdir("../kaggle")            # move into this folder
!pip install -q -r requirements.txt
!curl -fsSL https://get.docker.com | sh        # bootstrap docker daemon
!docker pull python:3.11-slim
from main import run
run(epochs=10, samples=50)       # auto-loads ../config/config.yaml
```

## Configuration highlights (`config/config.yaml`)

- **Dataset**: `dataset_name: mbpp` (or `humaneval`); switch by editing one key.
  MBPP gives per-test partial credit; HumanEval is all-or-nothing.
- **Model**: `model_name` / `ref_model_name` — defaults to **Qwen3-4B-Instruct-2507**
  (3B / 0.5B / 2B retired and commented out).
- **GPRO**: `gpro_num_samples_per_prompt` (group size for advantage)
  `gpro_clip_epsilon` `gpro_kl_coefficient` etc.
- **Sandbox**: `sandbox_image` `execution_timeout` `memory_limit_mb` `cpu_limit`.

## Notes / caveats

- The sandbox **requires a running Docker daemon**; it intentionally fails loudly
  if Docker is unavailable (no CPU fallback).
- `python:3.11-slim` only ships the stdlib. If a dataset needs numpy/scipy etc.,
  install them into the image once:
  ```bash
  docker run python:3.11-slim python -m pip install numpy
  docker commit $(docker ps -lq) python:3.11-slim
  ```