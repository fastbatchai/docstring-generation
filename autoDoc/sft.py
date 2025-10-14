import os
import pathlib

import modal

import autoDoc.dataset_utils as du
from autoDoc.common import (
    app,
    checkpoint_volume,
    dataset_cache_volume,
    model_cache_volume,
    sft_image,
)
from autoDoc.config import SFTExperimentConfig

########################################################
with sft_image.imports():
    # isort: off
    import unsloth
    import torch
    import wandb
    from datasets import load_dataset, load_from_disk
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

GPU_TYPE = "L40S"
TIMEOUT_HOURS = 6
MAX_RETRIES = 3


################
#  Dataset
#################
def load_or_cache_dataset(config, lang, seed):
    dataset_cache_path = (
        pathlib.Path("/dataset_cache")
        / "datasets"
        / config.dataset_name.replace("/", "--")
    )

    train_ds_path = dataset_cache_path / f"sft-train-{lang}-{config.max_samples}"

    if train_ds_path.exists():
        print(f"Loading cached dataset from {dataset_cache_path}")
        train_dataset = load_from_disk(
            dataset_cache_path / f"sft-train-{lang}-{config.max_samples}"
        )
        eval_dataset = load_from_disk(
            dataset_cache_path / f"sft-eval-{lang}-{config.max_samples}"
        )
    else:
        print(
            f"Downloading dataset: {config.dataset_name}, language: {lang}-{config.max_samples}"
        )

        ds = load_dataset(
            config.dataset_name, lang, split=f"train[:{config.max_samples}]"
        )
        ds = ds.map(
            lambda x: {"code": du.remove_first_docstring(x["func_code_string"])},
            batched=False,
        )

        ds = ds.train_test_split(test_size=1.0 - config.train_split_ratio, seed=seed)

        train_dataset = ds["train"].map(
            config.preproc_func,
            remove_columns=ds["train"].column_names,
            batched=False,
            fn_kwargs={"eos_token": config.eos_token},
        )
        eval_dataset = ds["test"].map(
            config.preproc_func,
            batched=False,
            remove_columns=[c for c in ds["test"].column_names if c != "language"],
            fn_kwargs={"eos_token": config.eos_token},
        )

        # Cache the processed datasets for future runs
        print(f"Caching processed datasets to {dataset_cache_path}")
        dataset_cache_path.mkdir(parents=True, exist_ok=True)
        train_dataset.save_to_disk(
            dataset_cache_path / f"sft-train-{lang}-{config.max_samples}"
        )
        eval_dataset.save_to_disk(
            dataset_cache_path / f"sft-eval-{lang}-{config.max_samples}"
        )

        # Commit the dataset cache to the volume
        dataset_cache_volume.commit()

    return train_dataset, eval_dataset


################
#  Model
#################
def load_or_cache_model(config):
    """Load or cache model"""
    print(f"Downloading and caching model: {config.model_name}")

    if config.use_unsloth:
        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_length,
            dtype=None,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            device_map="auto",
            token=os.environ["HF_TOKEN"],
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# The PEFT method is defined by the PeftConfig class that contains the parameters to build the PEFT model.
# For example, the LoRA method is defined by the LoraConfig,
def setup_model_for_finetuning_peft(config, model):
    if config.use_unsloth:
        return unsloth.FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,  # LoRA rank - higher values = more parameters
            target_modules=config.lora_target_modules,  # Which layers to apply LoRA to
            lora_alpha=config.lora_alpha,  # LoRA scaling parameter
            lora_dropout=config.lora_dropout,  # Dropout for LoRA layers
            bias=config.lora_bias,  # Bias configuration
            use_gradient_checkpointing=config.use_gradient_checkpointing,  # Memory optimization
            random_state=config.seed,  # Fixed seed for reproducibility
            use_rslora=False,  # Rank-stabilized LoRA
            loftq_config=None,  # LoFTQ quantization config
        )

    # 1- define the LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
    )
    # 2- create the PEFT model
    if config.load_in_4bit or config.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, lora_config)

    return model


def check_for_existing_checkpoint(checkpoint_dir):
    """
    Check if there's an existing checkpoint to resume training from.

    This enables resumable training, which is crucial for long-running experiments
    that might be interrupted by infrastructure issues or resource limits.

    taken from https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/unsloth_finetune.py#L549
    """

    if not checkpoint_dir.exists():
        return None

    # Look for the most recent checkpoint directory
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
        print(f"Found existing checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)

    return None


@app.function(
    image=sft_image,
    gpu=GPU_TYPE,
    volumes={
        "/model_cache": model_cache_volume,
        "/dataset_cache": dataset_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
    max_inputs=1,
)
def finetune(config):
    import time

    checkpoint_path = (
        pathlib.Path("/checkpoints") / "experiments" / config.experiment_name
    )

    wandb.init(
        project="docstring-finetune",
        name=config.experiment_name,
        config=config.__dict__,
    )

    print("Setting up model and data...")
    model, tokenizer = load_or_cache_model(config)
    train_dataset, eval_dataset = du.prepare_dataset(config, load_or_cache_dataset)

    if config.verbose > 1:
        for name, module in model.named_modules():
            if "proj" in name:
                print(name)

    # setup the model for finetuning
    model = setup_model_for_finetuning_peft(config, model)

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    resume_from_checkpoint = check_for_existing_checkpoint(checkpoint_path)
    print("Initialize training...")
    start_time = time.time()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if not config.skip_eval else None,
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=str(checkpoint_path),
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            max_length=config.max_length,
            seed=config.seed,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            eval_strategy="no" if config.skip_eval else "steps",
            save_strategy="steps",
            dataset_text_field="text",
            report_to="wandb",
            bf16=config.bf16,
        ),
    )

    if config.verbose > 0:
        print(f"Training dataset size: {len(train_dataset):,}")
        print(f"Evaluation dataset size: {len(eval_dataset):,}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )
        print(f"Experiment: {config.experiment_name}")

    if resume_from_checkpoint:
        print(f"Resuming training from {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("Starting training from scratch...")
        trainer.train()

    # Log custom metrics
    end_time = time.time()
    training_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

    wandb.log(
        {
            "total_training_time_seconds": training_time,
            "peak_memory_gb": peak_memory,
            "samples_per_second": len(train_dataset)
            * config.num_train_epochs
            / training_time,
            "time_per_epoch_seconds": training_time / config.num_train_epochs,
        }
    )
    print("Saving final model...")
    final_model_path = checkpoint_path / "final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Training completed! Model saved to: {final_model_path}")
    wandb.finish()


@app.local_entrypoint()
def main():
    config = SFTExperimentConfig()
    print(f"Starting finetuning experiment: {config.experiment_name}")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset.dataset_name}")
    # Launch the training job on Modal infrastructure
    finetune.remote(config)
    print(f"Training completed successfully: {config.experiment_name}")
