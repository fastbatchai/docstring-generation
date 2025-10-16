import json
import os
import pathlib

import modal

import autoDoc.dataset_utils as du
from autoDoc.common import checkpoint_volume, eval_app, eval_image
from autoDoc.config import EvaluationConfig


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
    import datasets
    import evaluate
    import numpy as np
    import torch
    from datasets import load_dataset
    from evaluate import logging
    from peft import PeftModel
    from torch.nn import CrossEntropyLoss
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from trl import HfPairwiseJudge


class Perplexity(evaluate.EvaluationModule):
    def _info(self):
        return evaluate.EvaluationModuleInfo(
            module_type="measurement",
            description="Modified version of perplexity. For more information, see https://huggingface.co/docs/transformers/perplexity",
            citation=None,
            inputs_description="model",
            features=datasets.Features(
                {
                    "data": datasets.Value("string"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )

    def _compute(
        self,
        data,
        model,
        tokenizer,
        batch_size: int = 16,
        add_start_token: bool = True,
        device=None,
    ):
        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], (
                "device should be either gpu or cpu."
            )
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(
                tokenizer.special_tokens_map_extended.values()
            )
            # check that the model already has at least one special token defined
            assert len(existing_special_tokens) > 0, (
                "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            )
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token:
            # leave room for <BOS> token to be added:
            assert tokenizer.bos_token is not None, (
                "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            )
            max_tokenized_len = model.config.max_length - 1
        else:
            max_tokenized_len = model.config.max_length

        encodings = tokenizer(
            data,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), (
                "Each input text must be at least one token long."
            )
        else:
            assert torch.all(torch.ge(attn_masks.sum(1), 2)), (
                "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."
            )

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
                ).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [
                        torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(
                            device
                        ),
                        attn_mask,
                    ],
                    dim=1,
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp2(
                (
                    loss_fct(shift_logits.transpose(1, 2), shift_labels)
                    * shift_attention_mask_batch
                ).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


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

    if config.training_type == "sft":
        preproc_fn = du.format_alpaca_example
    elif config.training_type == "grpo":
        preproc_fn = du.format_grpo_example

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

        dataset = dataset.map(lambda x: {"prompt": preproc_fn(x)}, batched=False)

        outputs = pipe(
            dataset["prompt"],
            batch_size=16,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        )
        generated_responses = [out[0]["generated_text"] for out in outputs]
        references = dataset["func_documentation_string"]
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
