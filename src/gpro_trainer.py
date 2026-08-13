import re
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
import mlflow
from typing import List, Dict, Any, Tuple, Optional
import torch.distributed as dist
from pathlib import Path

# ---------------------------------------------------------------------------
# Utility: setup HF Hub push
# ---------------------------------------------------------------------------
def _push_to_hub(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    experiment_name: str,
    run_name: str,
    epoch: int,
    overall_loss: float,
    config: Dict[str, Any],
    training_history: List[Dict[str, Any]],
):
    """Push model and training metadata to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi

        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("⚠️  HF_TOKEN not set – skipping Hub push.")
            return

        api = HfApi()
        repo_id = f"{experiment_name.lower().replace(' ', '-')}/{run_name.lower().replace(' ', '-')}-epoch{epoch}"

        # Create local repo directory
        repo_dir = f"./hf_repo_epoch_{epoch}"
        os.makedirs(repo_dir, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(repo_dir)
        tokenizer.save_pretrained(repo_dir)

        # Save training config and metrics
        training_meta = {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "epoch": epoch,
            "final_loss": overall_loss,
            "total_epochs": config.get("gpro_epochs", 10),
            "learning_rate": config.get("gpro_lr", 1e-5),
            "batch_size": config.get("gpro_batch_size", 4),
            "clip_epsilon": config.get("gpro_clip_epsilon", 0.2),
            "kl_coefficient": config.get("gpro_kl_coefficient", 0.01),
            "gae_lambda": config.get("gpro_gae_lambda", 0.95),
            "gamma": config.get("gpro_gamma", 0.99),
            "training_history": training_history,
        }
        with open(os.path.join(repo_dir, "training_metadata.json"), "w") as f:
            json.dump(training_meta, f, indent=2)

        # Create repo and push
        api.create_repo(repo_id, exist_ok=True, token=hf_token)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=repo_dir,
            commit_message=f"GPRO epoch {epoch} training run",
            token=hf_token,
        )

        print(f"🚀 Pushed model to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to push to HF Hub: {e}")


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class GproDataset(Dataset):
    def __init__(self, raw_data: List[Dict[str, Any]], tokenizer, max_new_tokens: int):
        self.prompts = []
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

        for item in raw_data:
            self.prompts.append(item["prompt"])

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.prompts[idx]
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_new_tokens,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs.input_ids.squeeze(0),  # [seq_len]
            "attention_mask": inputs.attention_mask.squeeze(0),  # [seq_len]
        }


# ---------------------------------------------------------------------------
# Roll-out generator
# ---------------------------------------------------------------------------
def rollout(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[str], List[float], List[float]]:
    """Returns (generated_codes, values, rewards)."""
    model.eval()
    generated_codes = []
    values = []
    rewards = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Extract last ```python ... ``` block
            code_blocks = re.findall(r"```python\s*(.*?)```", raw, re.DOTALL)
            code = code_blocks[-1].strip() if code_blocks else raw.strip()

            # Simple reward: 1 if "solution(" appears, else 0
            reward = 1.0 if "solution(" in code else 0.0

            # Dummy value head
            value = torch.randn(1).item()

            generated_codes.append(code)
            values.append(value)
            rewards.append(reward)

    return generated_codes, values, rewards


# ---------------------------------------------------------------------------
# GPRO Trainer Class
# ---------------------------------------------------------------------------
class GproTrainer:
    # Multiple policy configurations (keep one uncommented, others commented)
    # Policy 0: Default Qwen Coder
    # policy_0_name: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    # Policy 1: Alternative 1.5B
    # policy_1_name: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    # Policy 2: Open source alternative
    # policy_2_name: "meta-llama/Llama-3.2-1B-Instruct"

    def __init__(
        self,
        policy_model_name: str,
        ref_model_name: str,
        config: Dict[str, Any],
        device: str = "cuda"
        if torch.cuda.is_available()
        else "cpu",
    ):
        self.device = device
        self.config = config

        # --- Policy model (trainable) ---
        self.policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            policy_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).to(device)
        self.policy_model.train()

        # --- Reference model (frozen) ---
        self.ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).to(device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

# --- Optimizer ---
        gpro_lr = float(self.config.get("gpro_lr", 1e-5))
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=gpro_lr
        )

        # --- MLflow experiment ---
        mlflow.set_experiment(config.get("experiment_name", "RLVR_Group_Policy_Optimization"))

        # --- Checkpoint state ---
        self.checkpoint_dir = Path("./gpro_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
        self.training_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # GAE advantage estimation
    # ------------------------------------------------------------------
    def compute_advantages(self, rewards: List[float], values: List[float]) -> List[float]:
        advantages = [0.0] * len(rewards)
        running_adv = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.config.get("gpro_gamma", 0.99) * next_value - values[t]
            running_adv = delta + self.config.get("gpro_gae_lambda", 0.95) * running_adv
            advantages[t] = running_adv
        return advantages

    # ------------------------------------------------------------------
    # GPRO clipped surrogate loss
# ------------------------------------------------------------------
    # GPRO clipped surrogate loss
    # ------------------------------------------------------------------
    def _gpro_loss(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float,
    ) -> torch.Tensor:
        # Ensure same sequence length
        min_len = min(policy_logits.shape[1], ref_logits.shape[1])
        policy_logits = policy_logits[:, :min_len, :]
        ref_logits = ref_logits[:, :min_len, :]

        # Softmax & ratio – take max probability over vocabulary per token position
        policy_probs = F.softmax(policy_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)

        # Max probability per position: [batch, seq_len]
        policy_max, _ = policy_probs.max(dim=-1)
        ref_max, _ = ref_probs.max(dim=-1)

        # Average over sequence to get a single ratio per sample: [batch]
        ratio = (policy_max.mean(dim=-1) / (ref_max.mean(dim=-1) + 1e-8))

        # Clip surrogate
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

        # KL penalty term (optional)
        kl_penalty = self.config.get("gpro_kl_coefficient", 0.01) * (
            torch.log(ratio + 1e-8) - (-self.config.get("gpro_kl_target", 0.1))
        ).pow(2).mean()

        loss = -torch.min(surr1, surr2).mean() + kl_penalty
        return loss

    # ------------------------------------------------------------------
    # One full GPRO training iteration
    # ------------------------------------------------------------------
    def train(
        self,
        dataset: GproDataset,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        epochs = epochs or self.config.get("gpro_epochs", 10)
        batch_size = batch_size or self.config.get("gpro_batch_size", 4)

        # Custom collate for padding
        def collate_fn(batch):
            input_ids = [item["input_ids"] for item in batch]
            attention_masks = [item["attention_mask"] for item in batch]
            input_ids = pad_sequence(input_ids, batch_first=True)
            attention_masks = pad_sequence(attention_masks, batch_first=True)
            return {"input_ids": input_ids, "attention_mask": attention_masks}

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        self.policy_model.train()

        # Clip epsilon from config
        clip_epsilon = self.config.get("gpro_clip_epsilon", 0.2)

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # --- 1. Roll-out from policy ---
                prompts = self.policy_tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )
                codes, values, rewards = rollout(
                    self.policy_model,
                    self.policy_tokenizer,
                    prompts,
                    self.config.get("gpro_max_new_tokens", 256),
                    self.config.get("gpro_temperature", 0.7),
                    self.config.get("gpro_top_p", 1.0),
                )

                # --- 2. Compute advantages (GAE) per batch ---
                # rollout returns values/rewards for the entire prompts batch;
                # we compute GAE per DataLoader batch.
                batch_adv = []
                for i in range(0, len(values), batch_size):
                    batch_rewards = rewards[i : i + batch_size]
                    batch_values = values[i : i + batch_size]
                    adv = self.compute_advantages(batch_rewards, batch_values)
                    batch_adv.extend(adv)
                advantages = batch_adv

                # --- 3. Reference model forward (no grad) ---
                with torch.no_grad():
                    ref_inputs = self.ref_tokenizer(
                        prompts,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                    ).to(self.device)
                    ref_outputs = self.ref_model(
                        **ref_inputs, labels=ref_inputs["input_ids"]
                    )
                    ref_logits = ref_outputs.logits

                # --- 4. Policy model forward (with grad) ---
                policy_outputs = self.policy_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                policy_logits = policy_outputs.logits

                # --- 5. Compute GPRO loss ---
                loss = self._gpro_loss(
                    policy_logits,
                    ref_logits,
                    torch.tensor(advantages, dtype=torch.float32).to(self.device),
                    clip_epsilon,
                )

                # --- 6. Backpropagation ---
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            # --- Epoch summary ---
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            self.training_history.append(
                {"epoch": epoch, "avg_loss": avg_loss, "samples": len(dataset)}
            )
            mlflow.log_metric(
                f"gpro_epoch_{epoch}_loss", avg_loss, step=epoch
            )
            print(f"[GPRO] Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}")

            # --- Checkpoint saving (every gpro_checkpoint_interval epochs) ---
            checkpoint_interval = self.config.get(
                "gpro_checkpoint_interval", 5
            )
            if epoch % checkpoint_interval == 0 or epoch == epochs:
                ckpt_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "policy_state_dict": self.policy_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": avg_loss,
                        "config": self.config,
                    },
                    ckpt_path,
                )
                print(f"💾 Checkpoint saved: {ckpt_path}")

                # --- Push to HF Hub at checkpoints ---
                if self.config.get("log_artifacts", True):
                    try:
                        _push_to_hub(
                            self.policy_model,
                            self.policy_tokenizer,
                            self.config.get("experiment_name", "RLVR_Code_Generation"),
                            self.config.get("run_name", "Qwen_0.5B_RLPRO_Evaluation"),
                            epoch,
                            avg_loss,
                            self.config,
                            self.training_history,
                        )
                    except Exception as e:
                        print(f"⚠️  HF Hub push failed: {e}")

        # --- Final push after all epochs ---
        if self.config.get("log_artifacts", True):
            try:
                _push_to_hub(
                    self.policy_model,
                    self.policy_tokenizer,
                    self.config.get("experiment_name", "RLVR_Code_Generation"),
                    self.config.get("run_name", "Qwen_0.5B_RLPRO_Evaluation"),
                    epochs,
                    sum(h["avg_loss"] for h in self.training_history)
                    / len(self.training_history)
                    if self.training_history
                    else 0.0,
                    self.config,
                    self.training_history,
                )
            except Exception as e:
                print(f"⚠️  Final HF Hub push failed: {e}")

        # --- Final metrics ---
        final_loss = (
            sum(h["avg_loss"] for h in self.training_history)
            / len(self.training_history)
            if self.training_history
            else 0.0
        )
        mlflow.log_metric("gpro_final_loss", final_loss)

        return {
            "final_loss": final_loss,
            "epochs_trained": epochs,
            "samples_processed": len(dataset),
            "training_history": self.training_history,
        }


# ---------------------------------------------------------------------------
# Convenience wrapper: end-to-end GPRO cycle
# ---------------------------------------------------------------------------
def run_gpro_cycle(
    config_path: str = "config/config.yaml",
    epochs: Optional[int] = None,
    samples_limit: int = 100,
) -> Dict[str, Any]:
    """End-to-end GPRO training cycle configured from *config_path*."""
    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    epochs = epochs or cfg.get("gpro_epochs", 10)
    samples_limit = samples_limit or cfg.get("samples_limit", 100)

    # Load dataset
    from src.dataset import load_evaluation_dataset
    raw_data = load_evaluation_dataset()[:samples_limit]

    # Build dataset
    policy_tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    dataset = GproDataset(raw_data, policy_tok, cfg.get("gpro_max_new_tokens", 256))

    # Initialise trainer
    trainer = GproTrainer(
        policy_model_name=cfg["model_name"],
        ref_model_name=cfg.get("ref_model_name", cfg["model_name"]),
        config=cfg,
        device="cuda"
        if torch.cuda.is_available()
        else "cpu",
    )

    # Run training
    result = trainer.train(dataset=dataset, epochs=epochs, batch_size=cfg.get("gpro_batch_size", 4))

    mlflow.end_run()
    return result