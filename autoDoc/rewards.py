"""Reward functions for GRPO training."""

import re
from collections import Counter

import torch
from bert_score import score


def clean_text(text: str) -> str:
    """Clean text by removing special tokens and normalizing whitespace."""
    # Remove model-specific special tokens
    text = re.sub(r"<\|.*?\|>", "", text)

    # Remove excessive punctuation (optional, keep code identifiers if needed)
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def repetition_penalty(text: str) -> float:
    """Calculate repetition penalty as a score between 0 and 1.

    Higher values indicate more repetition (worse quality).
    """
    if not text.strip():
        return 1.0  # Maximum penalty for empty text

    tokens = text.split()
    if len(tokens) <= 1:
        return 0.0  # No repetition possible with 0-1 tokens

    counts = Counter(tokens)
    unique_tokens = len(counts)
    total_tokens = len(tokens)

    # Calculate repetition ratio: unique tokens / total tokens
    # Lower ratio = more repetition = higher penalty
    repetition_ratio = unique_tokens / total_tokens

    # Convert to penalty score (0 = no repetition, 1 = maximum repetition)
    penalty = 1.0 - repetition_ratio

    return min(penalty, 1.0)  # Ensure it's capped at 1.0


def length_penalty(docstring: str, optimal_range: tuple[int, int] = (50, 500)) -> float:
    """Penalize docstrings that are too short or too long.

    Args:
        docstring: Generated docstring
        optimal_range: (min_chars, max_chars) for optimal length

    Returns:
        Score between 0 and 1
    """
    if not docstring:
        return 0.0

    length = len(docstring.strip())
    min_len, max_len = optimal_range

    if length < min_len:
        return length / min_len
    elif length > max_len:
        return max(1.0 - (length - max_len) / max_len, 0.3)
    else:
        return 1.0


def token_overlap(pred: str, ref: str) -> float:
    """Calculate token overlap between prediction and reference."""
    pred_tokens = set(pred.split())
    ref_tokens = set(ref.split())
    return len(pred_tokens & ref_tokens) / max(len(ref_tokens), 1)


def length_reward(completions: list[str], **kwargs) -> list[float]:
    """Calculate length-based rewards for completions."""
    rewards = []
    for content in completions:
        rewards.append(length_penalty(content))
    return rewards


def semantic_reward(
    completions: list[str], docstring: list[str], **kwargs
) -> list[float]:
    """Calculate semantic reward combining BERTScore, token overlap, and repetition penalty."""
    clean_refs = [clean_text(r) for r in docstring]
    clean_preds = [clean_text(c) for c in completions]

    # Calculate BERTScore
    _, _, F1 = score(
        clean_preds,
        clean_refs,
        lang="en",
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    F1 = F1.tolist()

    rewards = []
    for i in range(len(clean_preds)):
        r_semantic = float(F1[i])
        r_overlap = float(token_overlap(clean_preds[i], clean_refs[i]))
        r_repetition = float(repetition_penalty(clean_preds[i]))

        # Ensure all components are between 0 and 1
        r_semantic = max(0.0, min(1.0, r_semantic))
        r_overlap = max(0.0, min(1.0, r_overlap))
        r_repetition = max(0.0, min(1.0, r_repetition))

        # Combined reward: semantic similarity + token overlap - repetition penalty
        # Use subtraction for repetition penalty since higher repetition = worse quality
        r_total = 0.6 * r_semantic + 0.3 * r_overlap - 0.1 * r_repetition

        # Ensure final reward is between 0 and 1
        r_total = max(0.0, min(1.0, r_total))
        rewards.append(r_total)

    return rewards
