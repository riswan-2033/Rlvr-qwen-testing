import yaml
import os
import multiprocessing
from src.dataset import load_evaluation_dataset
from src.model_runner  import LocalLLMRunner 
from src.sandbox import evaluate_in_isolation
from src.model_runner import LocalLLMRunner
from src.sandbox import evaluate_in_isolation
from src.tracker import MLflowPipelineTracker

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    # Enforce safe multiprocessing primitives on local workstation setups
    multiprocessing.freeze_support()
    
    config = load_config()
    
    # 1. Initialize Tracking Environment
    tracker = MLflowPipelineTracker(config["experiment_name"], config["run_name"])
    tracker.log_run_parameters(config)
    
    # 2. Load Evaluation Dataset 
    dataset = load_evaluation_dataset()
    
    # 3. Spin up local LLM Interface
    llm = LocalLLMRunner(config["model_name"])
    
    print("\n=== Beginning RLVR Execution Pass Pipeline ===")
    
    batch_size = config["batch_size"]
    
    # Slice iterations down to batches matching constraints
    for step_idx, i in enumerate(range(0, len(dataset), batch_size)):
        batch_items = dataset[i : i + batch_size]
        prompts = [item["prompt"] for item in batch_items]
        
        # Model generation phase
        generated_codes = llm.generate_code_solutions(
            prompts, 
            max_tokens=config["max_new_tokens"], 
            temp=config["temperature"]
        )
     
        # Verification and processing sequence loop
        for sample_idx, (item, generated_code) in enumerate(zip(batch_items, generated_codes)):
            global_sample_id = i + sample_idx
            
         

            # ... rest of main() stays the same ...

            # The sandbox is now called via:
            result = evaluate_in_isolation(
                code_string=generated_code,
                assertions=item["test_assertions"],
                timeout=config["execution_timeout"],
                memory_mb=config["memory_limit_mb"]
            )
                        
            print(f"Step {step_idx} | Sample {global_sample_id} -> Status: {result['status']} | Reward: {result['reward']}")
            
            # Record detailed structural parameters into MLflow UI
            tracker.log_evaluation_transaction(
                step_idx=step_idx,
                sample_id=global_sample_id,
                prompt=item["prompt"],
                generated_code=generated_code,
                verification_result=result
            )
            
    tracker.finish_tracking()
    print("\n=== Processing Complete. All trace logs mapped to MLflow dashboard. ===")

if __name__ == "__main__":
    main()
