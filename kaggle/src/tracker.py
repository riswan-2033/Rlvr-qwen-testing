# =============================================================================
# Comprehensive MLflow tracker for the RLVR code-verifier (Kaggle edition)
#
# Logs EVERYTHING so the MLflow UI tells the full story after completion:
#   - run params (full config)
#   - every rollout: input prompt, raw generated text, extracted code,
#     sandbox verdict, reward  (stored as trace artifacts + metrics)
#   - per-step & per-epoch metrics (loss, mean reward, pass rate)
#   - system metrics on every step (GPU util, GPU mem, CPU %, RAM)
#   - model checkpoints as artifacts in the run
# =============================================================================
import os
import json
import platform
import time
from typing import Dict, Any, List, Optional

import mlflow


def _log_metrics_many(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v), step=step)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  mlflow log_metric failed ({k}): {e}")


class MLflowCodeVerifierTracker:
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        config: Dict[str, Any],
        tracking_uri: Optional[str] = None,
    ):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name or "RLVR_Kaggle_Code")
        self.run = mlflow.start_run(run_name=run_name or "run")
        self.run_id = self.run.info.run_id

        # Log the full config as params + a JSON artifact.
        mlflow.log_params({str(k): str(v) for k, v in config.items()})
        self._save_json_artifact({"config": config}, "config/config.json")
        self._log_system_info()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _save_json_artifact(self, payload: Dict[str, Any], rel_path: str) -> None:
        tmp = f"mlflow_tmp_{int(time.time() * 1000)}.json"
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            mlflow.log_artifact(tmp, artifact_path=os.path.dirname(rel_path))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _log_system_info(self) -> None:
        info = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu_count": _gpu_count(),
            "gpu_names": _gpu_names(),
            "docker_host": os.environ.get("DOCKER_HOST", "local-socket"),
        }
        mlflow.set_tags(info)

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    def log_step_metrics(
        self,
        step: int,
        epoch: int,
        loss: float,
        mean_reward: float,
        pass_rate: float,
        gpu_util: float,
        gpu_mem_gb: float,
        cpu_percent: float,
        ram_gb: float,
    ) -> None:
        _log_metrics_many(
            {
                "epoch": epoch,
                "gpro_loss": loss,
                "mean_reward": mean_reward,
                "pass_rate": pass_rate,
                "system_gpu_util_pct": gpu_util,
                "system_gpu_mem_gb": gpu_mem_gb,
                "system_cpu_pct": cpu_percent,
                "system_ram_gb": ram_gb,
            },
            step=step,
        )

    def log_epoch_metrics(self, epoch: int, avg_loss: float, avg_reward: float) -> None:
        _log_metrics_many(
            {"epoch_avg_loss": avg_loss, "epoch_avg_reward": avg_reward}, step=epoch
        )

    def log_rollout(
        self,
        step: int,
        sample_idx: int,
        prompt: str,
        raw_text: str,
        code: str,
        verdict: Dict[str, Any],
        reward: float,
    ) -> None:
        """Log one generated sample: trace artifact + reward metric."""
        mlflow.log_metric(f"sample_{sample_idx}_reward", reward, step=step)
        payload = {
            "step": step,
            "sample_idx": sample_idx,
            "input_prompt": prompt,
            "raw_generated_text": raw_text,
            "extracted_code": code,
            "sandbox_status": verdict.get("status"),
            "sandbox_passed": verdict.get("passed"),
            "sandbox_total": verdict.get("total"),
            "sandbox_error": verdict.get("error"),
            "verifier_reward": verdict.get("reward"),
        }
        self._save_json_artifact(payload, f"rollouts/step_{step}/sample_{sample_idx}.json")

    def log_summary(self, history: List[Dict[str, Any]], epochs: int) -> None:
        self._save_json_artifact(
            {"epochs_trained": epochs, "history": history},
            "training/training_history.json",
        )
        if history:
            _log_metrics_many(
                {
                    "final_loss": history[-1].get("avg_loss", 0.0),
                    "final_reward": history[-1].get("avg_reward", 0.0),
                }
            )

    def log_checkpoint(self, ckpt_path: str, epoch: int) -> None:
        mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
        print(f"📦 MLflow artifact logged: {ckpt_path}")

    def finish(self) -> None:
        mlflow.end_run()


# ---------------------------------------------------------------------------
# System metrics helpers (psutil + pynvml, with graceful fallbacks)
# ---------------------------------------------------------------------------
def _gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return 0


def _gpu_names() -> List[str]:
    try:
        import torch

        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        return []


def get_system_metrics() -> Dict[str, float]:
    """Return current gpu/util, mem, cpu%, ram for MLflow logging."""
    out = {"gpu_util": 0.0, "gpu_mem_gb": 0.0, "cpu_pct": 0.0, "ram_gb": 0.0}

    # RAM
    try:
        import psutil

        vm = psutil.virtual_memory()
        out["ram_gb"] = round(vm.used / (1024 ** 3), 2)
        out["cpu_pct"] = psutil.cpu_percent(interval=None)
    except Exception:  # noqa: BLE001
        pass

    # GPU via pynvml (best) or torch fallback
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            util_vals, mem_vals = [], []
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(h)
                util_vals.append(util)
                mem_vals.append(mem_info.used / (1024 ** 3))
            out["gpu_util"] = round(sum(util_vals) / len(util_vals), 1)
            out["gpu_mem_gb"] = round(sum(mem_vals), 2)
        pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        try:
            import torch

            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / (1024 ** 3)
                alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
                out["gpu_mem_gb"] = round(mem, 2)
                out["gpu_peak_mem_gb"] = round(alloc, 2)
        except Exception:  # noqa: BLE001
            pass

    return out