# =============================================================================
# Kaggle RLVR Code Verifier - Entry point
#
#   docker pull python:3.11-slim        # once
#   python download_models.py            # once (optional pre-download)
#   python main.py --epochs 10 --samples 50
#
# EVERYTHING is logged to MLflow:
#   - all run config as params
#   - every rollout: input prompt, raw generated text, extracted code,
#     sandbox verdict + reward  (artifacts under rollouts/step_N/)
#   - per-step & per-epoch metrics + system metrics (GPU/CPU/RAM)
#   - model checkpoints as artifacts (checkpoints/)
# After training:  mlflow ui   → open the printed experiment/run URL
#
# For running inside a Kaggle notebook instead of CLI:
#   from main import run; run()
#
# CONFIG:
#   Uses the MAIN project config at ../config/config.yaml (relative to kaggle/).
#   Override with --config / config_path= if you place it elsewhere.
# =============================================================================
import os
import sys
import argparse
import yaml
from typing import Dict, Any, Optional


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config", "config.yaml")


def resolve_hf_home(cfg: Dict[str, Any]) -> None:
    """Point HF cache inside a Kaggle-visible dir so downloads persist."""
    cache = os.path.abspath(cfg.get("model_cache_dir", "kaggle_hf_cache/models"))
    os.makedirs(cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("HF_HUB_CACHE", cache)
    print(f"[HF] HF_HOME={os.environ['HF_HOME']}")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or DEFAULT_CONFIG
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def run(
    config_path: Optional[str] = None,
    epochs: Optional[int] = None,
    samples: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    resolve_hf_home(cfg)

    # Default MLflow backend to a sqlite DB in the working dir so `mlflow ui`
    # works cleanly on MLflow >=3 (file store is deprecated upstream).
    # Override anytime with MLFLOW_TRACKING_URI or cfg.mlflow_tracking_uri.
    cfg["mlflow_tracking_uri"] = (
        os.environ.get("MLFLOW_TRACKING_URI")
        or cfg.get("mlflow_tracking_uri")
        or "sqlite:///mlflow.db"
    )

    from src.dataset import load_code_dataset
    from src.gpro_trainer import GproTrainer

    # 1. Load coding dataset
    dataset = load_code_dataset(
        name=cfg.get("dataset_name", "mbpp"),
        split=cfg.get("dataset_split", "test"),
        limit=samples if samples is not None else cfg.get("dataset_sample_limit"),
    )
    print(f"[DATA] Loaded {len(dataset)} problems from {cfg.get('dataset_name')}")

    # 2. Trainer (policy + ref + Docker verifier reward)
    trainer = GproTrainer(
        policy_model_name=cfg["model_name"],
        ref_model_name=cfg.get("ref_model_name", cfg["model_name"]),
        config=cfg,
        cache_dir=cfg.get("model_cache_dir"),
    )

    # 3. Train (GPRO-style, Docker-sandboxed verifier rewards)
    result = trainer.train(dataset)

    print("\n=== Training complete ===")
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Kaggle RLVR code verifier")
    parser.add_argument("--config", default=None, help=f"(default: {DEFAULT_CONFIG})")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()
    run(args.config, epochs=args.epochs, samples=args.samples)


if __name__ == "__main__":
    main()