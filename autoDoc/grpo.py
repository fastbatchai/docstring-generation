"""Main training script for AutoDoc GRPO fine-tuning."""

import os
import pathlib
import sys

import modal

import autoDoc.dataset_utils as du
import autoDoc.rewards as rw
from autoDoc.common import app, checkpoint_volume, dataset_cache_volume, grpo_image
from autoDoc.config import DatasetConfig, GRPOExperimentConfig

with grpo_image.imports():
    import torch
    import wandb
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import login
    from trl import (
        GRPOConfig,
        GRPOTrainer,
        ModelConfig,
        ScriptArguments,
        TrlParser,
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
    )


# Uncomment to Upload the config file
# with dataset_cache_volume.batch_upload(force=True) as batch:
#     config_file_path = pathlib.Path(
#         "./data/config_grpo.yaml"
#     )
#     batch.put_file(
#         str(config_file_path),
#         "/grpo_config.yaml",
#     )


################
#  Dataset
#################
def load_or_cache_dataset(config, lang, seed):
    dataset_cache_path = (
        pathlib.Path("/dataset_cache")
        / "datasets"
        / config.dataset_name.replace("/", "--")
    )

    train_ds_path = dataset_cache_path / f"grpo-train-{lang}-{config.max_samples}"

    if train_ds_path.exists():
        print(f"Loading cached dataset from {dataset_cache_path}")
        train_dataset = load_from_disk(
            dataset_cache_path / f"grpo-train-{lang}-{config.max_samples}"
        )
        eval_dataset = load_from_disk(
            dataset_cache_path / f"grpo-eval-{lang}-{config.max_samples}"
        )
    else:
        print(f"Downloading dataset: {config.dataset_name}, language: {lang}")

        ds = load_dataset(
            config.dataset_name, lang, split=f"train[:{config.max_samples}]"
        )
        ds = ds.map(
            lambda x: {"code": du.remove_first_docstring(x["func_code_string"])},
            batched=False,
        )
        ds = ds.rename_column("func_documentation_string", "docstring")

        ds = ds.train_test_split(test_size=1.0 - config.train_split_ratio, seed=seed)

        train_dataset = ds["train"].map(
            config.preproc_func,
            batched=False,
        )
        eval_dataset = ds["test"].map(
            config.preproc_func,
            batched=False,
        )

        # Cache the processed datasets for future runs
        print(f"Caching processed datasets to {dataset_cache_path}")
        dataset_cache_path.mkdir(parents=True, exist_ok=True)
        train_dataset.save_to_disk(
            dataset_cache_path / f"grpo-train-{lang}-{config.max_samples}"
        )
        eval_dataset.save_to_disk(
            dataset_cache_path / f"grpo-eval-{lang}-{config.max_samples}"
        )

        # Commit the dataset cache to the volume
        dataset_cache_volume.commit()

    return train_dataset, eval_dataset


def finetune(config, training_args, model_args):
    import time

    checkpoint_path = (
        pathlib.Path("/checkpoints") / "experiments" / config.experiment_name
    )

    wandb.init(
        project="docstring-finetune",
        name=config.experiment_name,
        config=config.__dict__,
    )

    # prepare dataset
    train_dataset, test_dataset = du.prepare_dataset(config, load_or_cache_dataset)

    # prepapre trainer
    training_args.output_dir = checkpoint_path

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[rw.semantic_reward, rw.length_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
    )
    start_time = time.time()
    trainer.train()

    print(f"Training dataset size: {len(train_dataset):,}")
    print(f"Evaluation dataset size: {len(test_dataset):,}")
    print(f"Experiment: {config.experiment_name}")
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
    trainer.save_model(final_model_path)
    print(f"Training completed! Model saved to: {final_model_path}")
    wandb.finish()


@app.function(
    image=grpo_image,
    gpu="L40S",
    timeout=60 * 60 * 24,  # 24 hours
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={"/checkpoints": checkpoint_volume, "/dataset_cache": dataset_cache_volume},
)
def launch(args) -> None:
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))

    script_args, training_args, model_args = parser.parse_args_and_config(
        args=args, fail_with_unknown_args=False
    )
    print(f"Script args: {script_args}")

    ex_config = GRPOExperimentConfig(
        dataset=DatasetConfig(
            dataset_name=script_args.dataset_name,
            max_samples=1000,
            train_split_ratio=0.9,
        ),
        seed=training_args.seed,
        experiment_name="qwen-grpo-lora",
        wandb_project="docstring-finetune",
    )

    print(f"Training args: {training_args}")
    print(f"Model args: {model_args}")

    login(token=os.environ["HF_TOKEN"])

    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    finetune(ex_config, training_args, model_args)


@app.local_entrypoint()
def main(config):
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        sys.argv = sys.argv[4:]

    launch.remote(sys.argv)
    launch.remote(sys.argv)
