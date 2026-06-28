"""
ML classifier-based resume-to-function matching.

Trained on 2,484 real resumes from HuggingFace (opensporks/resumes).
Replaces the keyword-based O*NET matcher. Loads a pre-trained model
from disk — instant inference, no O*NET dependency, no API calls.
"""
from __future__ import annotations

import os
import pickle
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_HERE, "..", "data", "resume_classifier.pkl")
_CLASSES_PATH = os.path.join(_HERE, "..", "data", "resume_classifier_classes.json")

_model = None
_classes = None


def _load_model():
    global _model, _classes
    if _model is None:
        import json
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(_CLASSES_PATH) as f:
            _classes = json.load(f)
    return _model, _classes


def _clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def match_role_classifier(text: str) -> dict | None:
    """Classify a resume into the best-fit function category.

    Returns dict with: function, match_pct, alternatives, all_probas
    """
    model, classes = _load_model()
    cleaned = _clean(text)

    # Get probability for each class
    probas = model.predict_proba([cleaned])[0]

    # Sort by probability descending
    ranked = sorted(zip(classes, probas), key=lambda x: -x[1])

    best_func, best_prob = ranked[0]
    match_pct = round(best_prob * 100)

    alternatives = []
    for func, prob in ranked[1:]:
        pct = round(prob * 100)
        if pct >= 3:
            alternatives.append({
                "function": func,
                "match_pct": pct,
            })

    return {
        "function": best_func,
        "level": "Entry",
        "match_pct": match_pct,
        "alternatives": alternatives[:5],
        "all_probas": {func: round(p * 100, 1) for func, p in ranked[:10]},
    }
