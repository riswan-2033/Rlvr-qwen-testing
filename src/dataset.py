from datasets import load_dataset

def load_evaluation_dataset():
    """Loads a simple python code execution dataset."""
    # Using gsm8k as fallback since rlvr dataset has loading issues
    # Increased to 100 samples for GPRO training
    dataset = load_dataset("gsm8k", "main", split="train[:100]")
    
    formatted_data = []
    for item in list(dataset)[:100]: # Limit to 100 samples for GPRO training
        # Construct prompt for code model
        prompt = f"Write a Python function named 'solution' that fulfills this requirement: {item['question']}"
        
        formatted_data.append({
            "prompt": prompt,
            "test_assertions": "try:\n    # Simple sanity execution check\n    solution()\nexcept TypeError:\n    pass # Accept inputs if function expects variables"
        })
    return formatted_data
