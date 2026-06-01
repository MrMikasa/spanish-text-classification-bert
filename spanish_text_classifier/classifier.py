from __future__ import annotations

from functools import lru_cache
from typing import Callable, Iterable

DEFAULT_MODEL_ID = "finiteautomata/beto-sentiment-analysis"

LABELS_ES = {
    "POS": "Positivo",
    "NEU": "Neutral",
    "NEG": "Negativo",
    "LABEL_0": "Negativo",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positivo",
}


@lru_cache(maxsize=1)
def get_classifier(model_id: str = DEFAULT_MODEL_ID):
    from transformers import pipeline

    return pipeline("text-classification", model=model_id)


def classify_text(
    text: str,
    classifier: Callable[..., Iterable[dict]] | None = None,
    top_k: int = 3,
) -> list[dict]:
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Spanish text is required")

    pipe = classifier or get_classifier()
    output = pipe(clean_text, top_k=top_k, truncation=True)
    predictions = output[0] if output and isinstance(output[0], list) else output

    return [
        {
            "label": LABELS_ES.get(item["label"], item["label"]),
            "raw_label": item["label"],
            "score": round(float(item["score"]), 4),
        }
        for item in predictions
    ]
