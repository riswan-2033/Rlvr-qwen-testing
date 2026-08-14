# =============================================================================
# Pre-download the policy / reference models and tokenizer to a local cache.
# Run this FIRST in a Kaggle notebook so training starts without network stalls.
#
# Uses the MAIN project config: ../config/config.yaml (relative to kaggle/).
#
#   python download_models.py
# =============================================================================
import os
import yaml
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config", "config.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help=f"(default: {DEFAULT_CONFIG})")
    args = parser.parse_args()

    config_path = args.config or DEFAULT_CONFIG
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cache_dir = os.path.abspath(cfg.get("model_cache_dir", "kaggle_hf_cache/models"))
    os.makedirs(cache_dir, exist_ok=True)

    from huggingface_hub import snapshot_download

    names = set()
    names.add(cfg["model_name"])
    names.add(cfg.get("ref_model_name", cfg["model_name"]))

    for name in sorted(names):
        print(f"\n⏬ Downloading {name} -> {cache_dir}")
        snapshot_download(repo_id=name, cache_dir=cache_dir)
        print(f"✅ Done: {name}")


if __name__ == "__main__":
    main()