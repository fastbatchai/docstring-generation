"""Dataset processing utilities for docstring generation."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datasets import concatenate_datasets

from autoDoc.prompts import ALPACA_PROMPT, GRPO_PROMPT


def remove_first_docstring(code_str: str) -> str:
    """Remove first block comment or string literal inside a function.

    Handles Python triple quotes, Java/Go/JS/PHP block comments, and leading // lines.
    """
    # Python triple quotes (""" or ''')
    code_str = re.sub(r'("""|\'\'\')(.*?)\1', "", code_str, count=1, flags=re.DOTALL)
    # /** */ or /* */ comments
    code_str = re.sub(r"/\*\*(.*?)\*/", "", code_str, count=1, flags=re.DOTALL)
    code_str = re.sub(r"/\*(.*?)\*/", "", code_str, count=1, flags=re.DOTALL)
    # single-line // comments at the start
    code_str = re.sub(r"^\s*//.*?$", "", code_str, count=1, flags=re.MULTILINE)
    return code_str.strip()


def filter_quality_examples(example: dict[str, Any]) -> bool:
    """Filter out low-quality examples that might confuse the model."""
    code = example["func_code_string"]
    docstring = example["func_documentation_string"]

    # Skip examples with very short or very long code
    if len(code) < 50 or len(code) > 2000:
        return False

    # Skip examples with very short or missing docstrings
    if not docstring or len(docstring.strip()) < 10:
        return False

    return True


def format_alpaca_example(
    example: dict[str, Any], eos_token: str = None, mode="train"
) -> dict[str, str]:
    """
    Format example for instruction finetuning
    The output must be a dict for training
    """
    language = example["language"]
    code = example["code"]
    response = (
        example["docstring"] if example["docstring"] else "No docstring available."
    )
    if mode == "train":
        formatted_example = ALPACA_PROMPT.format(language, code, response) + eos_token
    else:
        formatted_example = ALPACA_PROMPT.format(language, code, "")
    return {"text": formatted_example}


def format_grpo_example(example: dict[str, Any]) -> dict[str, str]:
    """
    Format example for GRPO finetuning
    The output must be a dict for training
    The output must have the key `prompt`
    """
    language = example["language"]
    code = example["code"]

    formatted_example = GRPO_PROMPT.format(language=language, code=code)
    return {"prompt": formatted_example}


def prepare_dataset(config: dataclass, load_or_cache_dataset: Callable):
    print(f"Downloading and processing datasets: {config.dataset.dataset_name}")
    langs = ["python", "java", "javascript", "php", "ruby", "go"]

    train_datasets = []
    eval_datasets = []
    for lang in langs:
        train_ds, eval_ds = load_or_cache_dataset(
            config.dataset, lang, seed=config.seed
        )

        train_datasets.append(train_ds)
        eval_datasets.append(eval_ds)

    # Merge all into one dataset
    train_dataset = concatenate_datasets(train_datasets)
    eval_dataset = concatenate_datasets(eval_datasets)

    return train_dataset, eval_dataset
