import json
import os
import pathlib

import modal

import autoDoc.dataset_utils as du
from autoDoc.common import checkpoint_volume, eval_app, eval_image
from autoDoc.config import EvaluationConfig
from autoDoc.metrics import Perplexity


def calculate_exact_match(reference: str, hypothesis: str) -> bool:
    """Check if reference and hypothesis match exactly (ignoring whitespace)."""
    return reference.strip().lower() == hypothesis.strip().lower()


def calculate_token_overlap(reference: str, hypothesis: str) -> float:
    """Calculate token overlap (Jaccard similarity)."""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())

    if len(ref_tokens) == 0 and len(hyp_tokens) == 0:
        return 1.0

    intersection = ref_tokens.intersection(hyp_tokens)
    union = ref_tokens.union(hyp_tokens)

    return len(intersection) / len(union) if len(union) > 0 else 0.0


with eval_image.imports():
    import evaluate
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from trl import HfPairwiseJudge


@eval_app.function(
    image=eval_image,
    gpu="L40S",
    volumes={
        "/checkpoints": checkpoint_volume,
    },
    timeout=1 * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    max_inputs=1,  # Ensure we get a fresh container on retry
)
def evaluate_model(config):
    checkpoint_path = (
        pathlib.Path("/checkpoints")
        / "experiments"
        / config.experiment_name
        / "final_model"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        token=os.environ["HF_TOKEN"],
    )

    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, return_full_text=False
    )

    preproc_kwargs = {}
    if config.training_type == "sft":
        preproc_fn = du.format_alpaca_example
        preproc_kwargs = {"mode": "eval"}
        key = "text"
    elif config.training_type == "grpo":
        preproc_fn = du.format_grpo_example
        key = "prompt"

    bertscore = evaluate.load("bertscore")
    perplexity_metric = Perplexity()
    if config.use_llm_as_judge:
        judge = HfPairwiseJudge(model="meta-llama/Meta-Llama-3-70B-Instruct")

    all_results = {}
    all_metrics = {}
    all_references = {}
    # modal.interact()

    for lang in config.languages:
        print(f"\nEvaluating {lang}...")
        dataset = load_dataset(
            config.dataset_name, lang, split=f"test[:{config.num_samples}]"
        )
        dataset = dataset.map(
            lambda x: {"code": du.remove_first_docstring(x["func_code_string"])},
            batched=False,
        )
        dataset = dataset.rename_column("func_documentation_string", "docstring")

        dataset = dataset.map(preproc_fn, batched=False, fn_kwargs=preproc_kwargs)

        outputs = pipe(
            dataset[key],
            batch_size=16,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        )
        generated_responses = [out[0]["generated_text"] for out in outputs]
        references = dataset["docstring"]
        bert_results = bertscore.compute(
            predictions=generated_responses,
            references=references,
            lang="en",
            model_type="bert-base-uncased",
        )
        bert_f1 = sum(bert_results["f1"]) / len(bert_results["f1"])
        perplexities = perplexity_metric.compute(
            model=model, tokenizer=tokenizer, data=references
        )
        avg_perplexity = sum(perplexities["perplexities"]) / len(
            perplexities["perplexities"]
        )

        all_metrics[lang] = {
            "bertscore_f1": bert_f1,
            "perplexity": avg_perplexity,
        }

        if config.use_llm_as_judge:
            completions = [[c0, c1] for c0, c1 in zip(references, generated_responses)]
            best_idxs = judge.judge(dataset["prompt"], completions)
            model_win_rate = best_idxs.count(1) / len(best_idxs)
            all_metrics[lang]["model_win_rate"] = model_win_rate

        print(all_metrics[lang])

        all_results[lang] = generated_responses
        all_references[lang] = references
    return all_metrics, all_results, all_references


@eval_app.local_entrypoint()
def eval_main():
    config = EvaluationConfig()
    metrics, results, references = evaluate_model.remote(config)

    with open(
        f"{config.experiment_name}_generated_responses.json", "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    with open(f"{config.experiment_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    with open(f"{config.experiment_name}_references.json", "w", encoding="utf-8") as f:
        json.dump(references, f, indent=4, ensure_ascii=False)

    print(metrics)
