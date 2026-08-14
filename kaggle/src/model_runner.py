# =============================================================================
# Policy model rollout generation (Kaggle edition)
#
# The policy model is loaded with an EXPLICIT round-robin `device_map` (see
# `gpu_utils.build_multi_gpu_device_map`) so the layer stack is sharded across
# EVERY available GPU (2x T4 on Kaggle) - never parked on a single GPU.
# =============================================================================
import torch
import re
import concurrent.futures
from typing import List, Dict, Any, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalLLMRunner:
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        use_symlink_cache: bool = False,
    ):
        print(f"-> Loading policy model: {model_name}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required. CPU-only computation is not allowed.")
        self.num_gpus = torch.cuda.device_count()
        self.gpu_ids = list(range(self.num_gpus))
        print(f"[GPU] Sharding policy model across devices: {self.gpu_ids}")

        self.compute_dtype = (
            torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        from .gpu_utils import build_multi_gpu_device_map

        device_map = build_multi_gpu_device_map(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.compute_dtype,
            device_map=device_map,
            cache_dir=cache_dir,
        )
        self.model.eval()

    # ------------------------------------------------------------------
    @staticmethod
    def extract_code_block(raw: str) -> str:
        """Extract the last ```python ... ``` block from a model response."""
        blocks = re.findall(r"```python\s*(.*?)```", raw, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
        def_positions = [m.start() for m in re.finditer(r"def\s+\w+", raw)]
        if def_positions:
            remaining = raw[def_positions[-1]:]
            end_match = re.search(r"```", remaining)
            remaining = remaining[:end_match.start()] if end_match else remaining
            lines, code_lines = remaining.split("\n"), []
            for line in lines:
                if line.startswith("```"):
                    break
                if line.strip() == "" and code_lines and len(code_lines) > 5:
                    break
                code_lines.append(line)
            return "\n".join(code_lines).strip()
        return raw.strip()

    def _generate_batch(
        self,
        prompts: List[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> List[str]:
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

    def _extract_batch(self, raw_texts: List[str]) -> List[str]:
        return [self.extract_code_block(t) for t in raw_texts]

    def generate_code_solutions(
        self,
        prompts: List[str],
        temp: float,
        max_tokens: int = 256,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
    ) -> List[str]:
        """Generate code for every prompt; returns flattened list of code strings.

        If ``num_samples_per_prompt`` > 1, each prompt yields that many samples
        (multiple rollouts per prompt, used for the GPRO group advantage).
        """
        raw_texts = self.generate_code_solutions_raw(
            prompts, temp, max_tokens, top_p, num_samples_per_prompt
        )
        return self._extract_batch(raw_texts)

    def generate_code_solutions_raw(
        self,
        prompts: List[str],
        temp: float,
        max_tokens: int = 256,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
    ) -> List[str]:
        """Same as generate_code_solutions but returns the RAW model output text
        (before code-block extraction) so it can be logged in MLflow."""
        expanded = [p for p in prompts for _ in range(num_samples_per_prompt)]

        # Short-circuit tiny batches to keep throughput high.
        if len(expanded) <= self.num_gpus:
            return self._generate_batch(expanded, temp, max_tokens, top_p)

        # Fan out over GPUs with batched chunks to saturate both devices.
        chunk_size = max(1, len(expanded) // (self.num_gpus * 2))
        chunks = [
            expanded[i : i + chunk_size]
            for i in range(0, len(expanded), chunk_size)
        ]
        results = [None] * len(chunks)

        def worker(idx, prompt_chunk):
            results[idx] = self._generate_batch(prompt_chunk, temp, max_tokens, top_p)

        # Model.generate already uses all GPUs per-call (device_map sharding);
        # running chunk calls concurrently keeps both GPUs saturated.
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(4, self.num_gpus * 2)) as pool:
            futures = [
                pool.submit(worker, i, c) for i, c in enumerate(chunks)
            ]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        flat = [t for chunk in results for t in chunk]
        torch.cuda.empty_cache()
        return flat