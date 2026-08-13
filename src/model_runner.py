import re
import torch
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLMRunner:
    def __init__(self, model_name: str):
        print(f"-> Allocating VRAM and initializing weights for: {model_name}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required. CPU-only computation is not allowed.")
        self.num_gpus = torch.cuda.device_count()
        self.gpu_ids = list(range(self.num_gpus))
        self.device = f"cuda:{self.gpu_ids[0]}" if self.num_gpus > 0 else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # FP16 on Turing T4; BF16 requires Ampere+.
        dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
        self.fp_dtype = dtype

        # Load one full model copy per GPU so generation uses every GPU in parallel.
        self.models = {}
        for gpu in self.gpu_ids:
            print(f"[GPU {gpu}] Loading model replica on cuda:{gpu}")
            self.models[gpu] = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype
            ).to(f"cuda:{gpu}")
        self.model = self.models[self.gpu_ids[0]]

    def _generate_single_gpu(self, gpu: int, prompts: list, temp: float, max_tokens: int) -> list:
        model = self.models[gpu]
        generated = []
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(f"cuda:{gpu}")
                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                raw_response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                code_blocks = re.findall(r'```python\s*(.*?)```', raw_response, re.DOTALL)
                if code_blocks:
                    cleaned_code = code_blocks[-1].strip()
                else:
                    def_positions = [m.start() for m in re.finditer(r'def\s+\w+', raw_response)]
                    if def_positions:
                        last_def = def_positions[-1]
                        remaining = raw_response[last_def:]
                        end_match = re.search(r'```', remaining)
                        if end_match:
                            cleaned_code = remaining[:end_match.start()].strip()
                        else:
                            lines = remaining.split('\n')
                            code_lines = []
                            for line in lines:
                                if line.startswith('```'):
                                    break
                                if line.strip() == '' and code_lines and len(code_lines) > 5:
                                    break
                                code_lines.append(line)
                            cleaned_code = '\n'.join(code_lines).strip()
                    else:
                        cleaned_code = raw_response.strip()
                generated.append(cleaned_code)
        return generated

    def generate_code_solutions(self, prompts: list, temp: float, max_tokens=526) -> list:
        """Split prompts across all GPUs and generate in parallel."""
        generated_outputs = [None] * len(prompts)

        if self.num_gpus < 2 or len(prompts) < 2:
            return self._generate_single_gpu(self.gpu_ids[0], prompts, temp, max_tokens)

        # Partition prompts across GPUs.
        chunks = {gpu: [] for gpu in self.gpu_ids}
        for idx, prompt in enumerate(prompts):
            chunks[self.gpu_ids[idx % self.num_gpus]].append((idx, prompt))

        def run_gpu(gpu):
            return gpu, self._generate_single_gpu(
                gpu, [p for _, p in chunks[gpu]], temp, max_tokens
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as pool:
            for gpu, results in pool.map(run_gpu, self.gpu_ids):
                for (orig_idx, _), code in zip(chunks[gpu], results):
                    generated_outputs[orig_idx] = code

        torch.cuda.empty_cache()
        return generated_outputs