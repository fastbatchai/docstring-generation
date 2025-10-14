import modal

base_image: modal.Image = (
    modal.Image.debian_slim()
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "trl==0.19.1",
        "transformers==4.54.0",
        "wandb==0.21.0",
        "bitsandbytes",
        "bert-score",
        "ipython",
    )
    .env({"HF_HOME": "/model_cache"})
)

# Extend base image with GRPO dependencies
grpo_image: modal.Image = base_image.uv_pip_install(
    "bert-score",
).add_local_python_source("autoDoc")

sft_image: modal.Image = base_image.uv_pip_install(
    "unsloth[cu128-torch270]==2025.7.8",
    "unsloth_zoo==2025.7.10",
).add_local_python_source("autoDoc")

app = modal.App(
    "docstring-finetune",
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)


# Modal volumes to store checkpoints, dataset cache, and model cache
checkpoint_volume = modal.Volume.from_name(
    "finetune-checkpoints", create_if_missing=True
)
dataset_cache_volume = modal.Volume.from_name(
    "finetune-dataset-cache", create_if_missing=True
)
model_cache_volume = modal.Volume.from_name(
    "finetune-model-cache", create_if_missing=True
)
