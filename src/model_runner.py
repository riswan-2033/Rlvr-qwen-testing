import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLMRunner:
    def __init__(self, model_name: str):
        print(f"-> Allocating VRAM and initializing weights for: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use bfloat16 for your Ampere RTX 3060 architecture to preserve stability
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def generate_code_solutions(self, prompts: list, max_tokens: int, temp: float) -> list:
        """Inference sequence optimized for strict single-GPU constraint setups."""
        generated_outputs = []
        
        # Enforce inference safety flags to stop graph allocation calculations
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                output_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract solution block - prefer the last code block
                raw_response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                # Find all code blocks and take the last one (usually most complete)
                code_blocks = re.findall(r'```python\s*(.*?)```', raw_response, re.DOTALL)
                if code_blocks:
                    cleaned_code = code_blocks[-1].strip()
                else:
                    # Fallback: find the last def and extract code from there
                    def_positions = [m.start() for m in re.finditer(r'def\s+\w+', raw_response)]
                    if def_positions:
                        last_def = def_positions[-1]
                        remaining = raw_response[last_def:]
                        # Find closing ``` or take substantial portion
                        end_match = re.search(r'```', remaining)
                        if end_match:
                            cleaned_code = remaining[:end_match.start()].strip()
                        else:
                            # Take lines until we hit explanatory text
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
                generated_outputs.append(cleaned_code)
                
        # Proactively purge VRAM fragments before executing code evaluation passes
        torch.cuda.empty_cache()
        return generated_outputs