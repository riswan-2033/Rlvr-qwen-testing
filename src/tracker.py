import mlflow
import json
import os

class MLflowPipelineTracker:
    def __init__(self, experiment_name: str, run_name: str):
        mlflow.set_experiment(experiment_name)
        self.active_run = mlflow.start_run(run_name=run_name)

    def log_run_parameters(self, configurations: dict):
        mlflow.log_params(configurations)

    def log_evaluation_transaction(self, step_idx: int, sample_id: int, prompt: str, generated_code: str, verification_result: dict):
        """Logs comprehensive input-output metadata arrays directly into MLflow."""
        # Log aggregated numerical score metrics
        mlflow.log_metric(f"reward_sample_{sample_id}", verification_result["reward"], step=step_idx)
        
        # Save exact transaction data trace structures as tracking artifacts
        transaction_payload = {
            "step": step_idx,
            "sample_id": sample_id,
            "input_prompt": prompt,
            "generated_output_code": generated_code,
            "sandbox_status": verification_result["status"],
            "sandbox_error": verification_result["error"],
            "reward_assigned": verification_result["reward"]
        }
        
        artifact_filename = f"trace_step_{step_idx}_sample_{sample_id}.json"
        with open(artifact_filename, "w") as f:
            json.dump(transaction_payload, indent=4, fp=f)
            
        mlflow.log_artifact(artifact_filename, artifact_path="execution_traces")
        os.remove(artifact_filename) # Maintain local disk space cleanup

    def finish_tracking(self):
        mlflow.end_run()
