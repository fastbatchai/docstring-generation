"""Configuration management for AutoDoc GRPO training."""

from dataclasses import dataclass, field
from typing import Callable

from autoDoc.dataset_utils import format_alpaca_example, format_grpo_example


@dataclass
class DatasetConfig:
    """Dataset configuration"""

    dataset_name: str = "claudios/code_search_net"
    max_samples: int = 1000
    train_split_ratio: float = 0.9
    preproc_func: Callable = None
    eos_token: str = "<|endoftext|>"
    training_type: str = ""


@dataclass
class GRPOExperimentConfig:
    """Complete experiment configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    seed: int = 42
    wandb_project: str = ""
    experiment_name: str = ""

    def __post_init__(self):
        self.dataset.preproc_func = format_grpo_example
        self.dataset.training_type = "grpo"


@dataclass
class SFTExperimentConfig:
    """Configuration for instruction finetuning experiment."""

    # Dataset Configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    seed: int = 42
    verbose: int = 0

    # Model Configuration
    model_name: str = "google/codegemma-2b"  # "bigcode/starcoder2-3b" #"meta-llama/Llama-3.2-3B-Instruct" #"Qwen/Qwen2.5-Coder-1.5B-Instruct"
    max_length: int = 1024
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # LoRA Configuration
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )  # , "k_proj", "o_proj","gate_proj","up_proj","down_proj",
    lora_bias: str = "none"

    # Training Configuration
    experiment_name: str = "codegemma-lora-unsloth-1"
    num_train_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    warmup_ratio: float = 0.03
    fp16: bool = False
    bf16: bool = True
    use_unsloth: bool = True
    use_gradient_checkpointing: str = "unsloth"

    # Logging and Evaluation
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 3
    skip_eval: bool = False
    wandb_project: str = "docstring-finetune"
    report_to: str = "wandb"

    def __post_init__(self):
        self.dataset.preproc_func = format_alpaca_example
        self.dataset.training_type = "sft"
