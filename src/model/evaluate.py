"""
evaluate.py
Compute WER (Word Error Rate) and CER (Character Error Rate).
"""

from typing import List
import editdistance


def _tokenize(text: str) -> List[str]:
    return text.strip().lower().split()


def wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate = edit_distance(ref_words, hyp_words) / len(ref_words).

    Args:
        reference:  Ground-truth transcript.
        hypothesis: Model prediction.

    Returns:
        WER as a float in [0, ∞).
    """
    ref_words = _tokenize(reference)
    hyp_words = _tokenize(hypothesis)
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float("inf")
    return editdistance.eval(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate = edit_distance(ref_chars, hyp_chars) / len(ref_chars).

    Args:
        reference:  Ground-truth transcript.
        hypothesis: Model prediction.

    Returns:
        CER as a float in [0, ∞).
    """
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else float("inf")
    return editdistance.eval(ref_chars, hyp_chars) / len(ref_chars)


def evaluate_dataset(references: List[str], hypotheses: List[str]) -> dict:
    """
    Compute average WER and CER over a dataset.

    Args:
        references:  List of ground-truth transcripts.
        hypotheses:  List of model predictions.

    Returns:
        {'wer': float, 'cer': float}
    """
    assert len(references) == len(hypotheses), "Length mismatch."
    avg_wer = sum(wer(r, h) for r, h in zip(references, hypotheses)) / len(references)
    avg_cer = sum(cer(r, h) for r, h in zip(references, hypotheses)) / len(references)
    return {"wer": avg_wer, "cer": avg_cer}
