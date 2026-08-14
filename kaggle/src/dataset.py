# =============================================================================
# Dataset loading for the RLVR code-verifier (Kaggle edition)
#
# Supported datasets (switch via config.dataset_name):
#   - mbpp       : Mostly Basic Python Problems (standard coding benchmark)
#   - humaneval  : OpenAI HumanEval (function-completion benchmark)
#
# Each returned item has:
#   prompt            : the text sent to the policy LLM
#   tests             : list of assertion strings (already function-name aware)
#   setup_code        : optional imports/helpers needed before tests run
#   entry_point       : name of the function the model must produce
# =============================================================================
import re
import os
from typing import List, Dict, Any, Optional


def _instruct_prompt(task: str, entry_point: str) -> str:
    """Wrap a coding task into an instruction prompt."""
    return (
        "You are an expert Python programmer.\n"
        "Write a Python function named '"
        + entry_point
        + "' that solves the problem below.\n"
        "Only return the function definition in a single ```python code block.\n"
        "Do not include tests, explanations, or extra output.\n\n"
        "PROBLEM:\n"
        + task.strip()
    )


def _extract_entry_point(code: str) -> str:
    """Pull the function name from reference code, defaulting to 'solution'."""
    m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    return m.group(1) if m else "solution"


# -----------------------------------------------------------------------------
# MBPP
# -----------------------------------------------------------------------------
def _load_mbpp(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("mbpp", split=split)
    items = []
    for row in list(ds)[:limit]:
        entry_point = _extract_entry_point(row["code"])
        items.append(
            {
                "prompt": _instruct_prompt(row["text"], entry_point),
                "tests": list(row["test_list"]),
                "setup_code": row.get("test_setup_code") or "",
                "entry_point": entry_point,
                "task_id": str(row.get("task_id", "")),
            }
        )
    return items


# -----------------------------------------------------------------------------
# HumanEval
# -----------------------------------------------------------------------------
def _load_humaneval(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    # HumanEval has a single 'test' split; split name is ignored.
    ds = load_dataset("openai/openai_humaneval", split="test")
    items = []
    for row in list(ds)[:limit]:
        entry_point = row["entry_point"]
        # HumanEval prompt already contains the function signature; the model
        # must continue the code. 'test' calls check(entry_point).
        items.append(
            {
                "prompt": row["prompt"],  # raw prompt (model completes it)
                "tests": [row["test"]],
                "setup_code": "",
                "entry_point": entry_point,
                "task_id": row["task_id"],
            }
        )
    return items


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------
def load_code_dataset(
    name: str = "mbpp",
    split: str = "test",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load a coding dataset; returns list of {prompt, tests, setup_code, entry_point}."""
    name = (name or "mbpp").strip().lower()
    if name in ("mbpp", "mostly_basic_python_problems"):
        return _load_mbpp(split, limit)
    if name in ("humaneval", "openai_humaneval", "openai/humaneval"):
        return _load_humaneval(split, limit)
    raise ValueError(f"Unknown dataset: {name}. Use 'mbpp' or 'humaneval'.")
