# =============================================================================
# Self-contained GPRO-style trainer with the Docker verifier as reward source
#
# Training loop per step:
#   1. Sample `gpro_batch_size` prompts from the dataset.
#   2. Generate `gpro_num_samples_per_prompt` code rollouts per prompt.
#   3. Run every rollout in the Docker sandbox -> verifier reward in [0, 1].
#   4. Build group advantages (reward - group mean, / group std) per prompt.
#   5. Compute policy vs reference log-prob ratios on the generated tokens.
#   6. GPRO clipped surrogate loss + KL penalty -> backprop -> optimizer step.
# =============================================================================
import os
import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from .model_runner import LocalLLMRunner
from .sandbox import evaluate_in_isolation
from .tracker import MLflowCodeVerifierTracker, get_system_metrics


class GproTrainer:
    def __init__(
        self,
        policy_model_name: str,
        ref_model_name: str,
        config: Dict[str, Any],
        cache_dir: Optional[str] = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required. CPU-only computation is not allowed.")
        self.config = config
        self.num_gpus = torch.cuda.device_count()
        self.gpu_ids = list(range(self.num_gpus))
        self.device = torch.device("cuda:0")
        self.compute_dtype = (
            torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
        )
        self.policy_runner = LocalLLMRunner(
            policy_model_name, cache_dir=cache_dir,
            use_symlink_cache=self.config.get("use_symlink_cache", False),
        )
        self.policy_model = self.policy_runner.model
        self.policy_tokenizer = self.policy_runner.tokenizer
        self.policy_model.train()

        # Frozen reference model for KL regularisation.
        print(f"-> Loading reference model: {ref_model_name}")
        self.ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_name, cache_dir=cache_dir)
        if self.ref_tokenizer.pad_token is None:
            self.ref_tokenizer.pad_token = self.ref_tokenizer.eos_token
        from .gpu_utils import build_multi_gpu_device_map

        ref_device_map = build_multi_gpu_device_map(ref_model_name, cache_dir=cache_dir)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_name,
            torch_dtype=self.compute_dtype,
            device_map=ref_device_map,
            cache_dir=cache_dir,
        )
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=float(self.config.get("gpro_lr", 1e-5)),
        )
        self.clip_epsilon = float(self.config.get("gpro_clip_epsilon", 0.2))
        self.kl_coeff = float(self.config.get("gpro_kl_coefficient", 0.01))
        self.checkpoint_dir = self.config.get("checkpoint_dir", "kaggle_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- MLflow tracker (logs everything) ---
        from .tracker import MLflowCodeVerifierTracker
        self.tracker = MLflowCodeVerifierTracker(
            experiment_name=self.config.get("experiment_name", "RLVR_Kaggle_Code"),
            run_name=self.config.get("run_name", "run"),
            config=self.config,
            tracking_uri=self.config.get("mlflow_tracking_uri"),
        )

    # ------------------------------------------------------------------
    # Log-probabilities of generated text
    # ------------------------------------------------------------------
    def _completion_logprobs(
        self, model, tokenizer, texts: List[str], use_grad: bool = False
    ) -> torch.Tensor:
        """Per-sample sum of log-probs over the generated (non-padding) tokens.

        Set ``use_grad=True`` for the POLICY model so gradients flow into it.
        Keep ``use_grad=False`` for the frozen reference model.
        """
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        if use_grad:
            out = model(**encoded)
        else:
            with torch.no_grad():
                out = model(**encoded)
        logits = out.logits.float()  # [B, T, V]
        logprobs = F.log_softmax(logits, dim=-1)
        shift_logprobs = logprobs[:, :-1, :]
        shift_ids = encoded["input_ids"][:, 1:]
        per_token = shift_logprobs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
        mask = encoded["attention_mask"][:, 1:].float()
        return (per_token * mask).sum(-1)

    # ------------------------------------------------------------------
    # Group advantages (GRPO-style baseline within the prompt group)
    # ------------------------------------------------------------------
    def _group_advantages(self, rewards: List[float], group_size: int) -> torch.Tensor:
        advs = []
        for start in range(0, len(rewards), group_size):
            group = rewards[start : start + group_size]
            mean = sum(group) / len(group)
            std = (sum((r - mean) ** 2 for r in group) / len(group)) ** 0.5
            std = std if std > 1e-8 else 1.0
            advs.extend((r - mean) / std for r in group)
        return torch.tensor(advs, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # GPRO clipped surrogate loss
    # ------------------------------------------------------------------
    def _gpro_loss(self, pol_logp: torch.Tensor, ref_logp: torch.Tensor,
                   advantages: torch.Tensor) -> torch.Tensor:
        ratio = torch.exp(pol_logp - ref_logp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        kl = (pol_logp - ref_logp).pow(2).mean()
        loss = -torch.min(surr1, surr2).mean() + self.kl_coeff * kl
        return loss

    # ------------------------------------------------------------------
    # One full training run
    # ------------------------------------------------------------------
    def train(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        epochs = int(self.config.get("gpro_epochs", 10))
        batch_size = int(self.config.get("gpro_batch_size", 4))
        samples_per_prompt = int(self.config.get("gpro_num_samples_per_prompt", 4))
        max_tokens = int(self.config.get("gpro_max_new_tokens", 256))
        temperature = float(self.config.get("gpro_temperature", 0.7))
        top_p = float(self.config.get("gpro_top_p", 1.0))
        image = self.config.get("sandbox_image", "python:3.11-slim")
        timeout = float(self.config.get("execution_timeout", 5.0))
        memory_mb = int(self.config.get("memory_limit_mb", 512))
        cpu_limit = self.config.get("cpu_limit", "0.5")
        network_disabled = bool(self.config.get("network_disabled", True))
        read_only_fs = bool(self.config.get("read_only_fs", True))

        from .sandbox import get_client
        docker_client = get_client()

        step = 0
        history = []
        for epoch in range(1, epochs + 1):
            epoch_losses, epoch_rewards = [], []
            for start in range(0, len(dataset), batch_size):
                batch_items = dataset[start : start + batch_size]
                prompts = [it["prompt"] for it in batch_items]
                group_size = samples_per_prompt * len(prompts)

                # --- 1+2. Generate rollouts (raw text kept for MLflow) ------
                raw_texts = self.policy_runner.generate_code_solutions_raw(
                    prompts, temperature, max_tokens, top_p, num_samples_per_prompt=samples_per_prompt
                )
                codes = [self.policy_runner.extract_code_block(t) for t in raw_texts]

                # --- 3. Verifier (Docker sandbox) ---------------------------
                rewards = []
                verdicts = []
                flat_items = [it for it in batch_items for _ in range(samples_per_prompt)]
                for it, code in zip(flat_items, codes):
                    result = evaluate_in_isolation(
                        code=code,
                        tests=it["tests"],
                        setup_code=it["setup_code"],
                        image=image,
                        timeout=timeout,
                        memory_mb=memory_mb,
                        cpu_limit=cpu_limit,
                        network_disabled=network_disabled,
                        read_only_fs=read_only_fs,
                        client=docker_client,
                    )
                    rewards.append(result["reward"])
                    verdicts.append(result)

                # --- Log every rollout to MLflow (prompt/raw/code/verdict) --
                flat_prompts = [p for p in prompts for _ in range(samples_per_prompt)]
                for idx, (it, prompt, raw, code, verdict, reward) in enumerate(
                    zip(flat_items, flat_prompts, raw_texts, codes, verdicts, rewards)
                ):
                    self.tracker.log_rollout(
                        step=step, sample_idx=(start + idx),
                        prompt=prompt, raw_text=raw, code=code,
                        verdict=verdict, reward=reward,
                    )

                # --- 4. Advantages ------------------------------------------
                advantages = self._group_advantages(rewards, group_size)

                # --- 5+6. Log-probs + loss on full generated texts ----------
                full_texts = [
                    f"{prompt}\n```python\n{code}\n```"
                    for prompt, code in zip(
                        [p for p in prompts for _ in range(samples_per_prompt)], codes
                    )
                ]
                self.policy_model.train()
                pol_logp = self._completion_logprobs(
                    self.policy_model, self.policy_tokenizer, full_texts, use_grad=True
                )
                ref_logp = self._completion_logprobs(self.ref_model, self.ref_tokenizer, full_texts)

                loss = self._gpro_loss(pol_logp, ref_logp.detach(), advantages)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                self.optimizer.step()

                step += 1
                epoch_losses.append(loss.item())
                epoch_rewards.extend(rewards)
                mean_r = sum(rewards) / len(rewards)

                # --- System metrics (GPU/CPU/RAM) on every step -------------
                sys_metrics = get_system_metrics()
                self.tracker.log_step_metrics(
                    step=step, epoch=epoch,
                    loss=loss.item(), mean_reward=mean_r, pass_rate=mean_r,
                    gpu_util=sys_metrics["gpu_util"],
                    gpu_mem_gb=sys_metrics["gpu_mem_gb"],
                    cpu_percent=sys_metrics["cpu_pct"],
                    ram_gb=sys_metrics["ram_gb"],
                )

                print(
                    f"[GPRO] Epoch {epoch} | step {step} | loss {loss.item():.4f} | "
                    f"mean reward (verifier) {mean_r:.3f} | pass rate {(mean_r):.1%} | "
                    f"GPU util {sys_metrics['gpu_util']}% | "
                    f"GPU mem {sys_metrics['gpu_mem_gb']}GB"
                )

            avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            avg_reward = sum(epoch_rewards) / max(1, len(epoch_rewards))
            history.append({"epoch": epoch, "avg_loss": avg_loss, "avg_reward": avg_reward})
            self.tracker.log_epoch_metrics(epoch, avg_loss, avg_reward)

            if epoch % int(self.config.get("checkpoint_interval", 1)) == 0:
                ckpt = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.ckpt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.policy_runner.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": history,
                        "config": self.config,
                    },
                    ckpt,
                )
                self.tracker.log_checkpoint(ckpt, epoch)
                print(f"💾 Checkpoint saved + logged: {ckpt}")

            # Push per-epoch to HF Hub if requested + token present
            if self.config.get("log_artifacts") and os.environ.get(self.config.get("hf_token_env", "HF_TOKEN")):
                try:
                    self._push_epoch(epoch, avg_loss, avg_reward)
                except Exception as e:  # noqa: BLE001
                    print(f"⚠️  HF push failed (epoch {epoch}): {e}")

        # --- Final summary in MLflow ----------------------------------------
        self.tracker.log_summary(history, epochs)
        self.tracker.finish()

        return {"history": history, "epochs": epochs}

    def _push_epoch(self, epoch: int, loss: float, reward: float) -> None:
        from huggingface_hub import HfApi
        from pathlib import Path
        import json

        token = os.environ.get(self.config.get("hf_token_env", "HF_TOKEN"))
        api = HfApi()
        repo_dir = Path("kaggle_hf_export") / f"epoch_{epoch}"
        repo_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = (
            self.policy_runner.model.module
            if hasattr(self.policy_runner.model, "module")
            else self.policy_runner.model
        )
        model_to_save.save_pretrained(repo_dir)
        self.policy_tokenizer.save_pretrained(repo_dir)
        (repo_dir / "metrics.json").write_text(json.dumps({"loss": loss, "reward": reward}))
        repo_id = f"{self.config.get('experiment_name','RLVR').lower().replace(' ','-')}/policy-epoch{epoch}"
        api.create_repo(repo_id, exist_ok=True, token=token)
        api.upload_folder(repo_id=repo_id, folder_path=str(repo_dir), token=token)
        print(f"🚀 Pushed epoch {epoch} to https://huggingface.co/{repo_id}")