"""Inferencia local; nunca deserializar modelos proporcionados por un cliente."""

from functools import lru_cache
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "newsscope.joblib"
CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")


@lru_cache(maxsize=1)
def load_model():
    """Carga una sola vez el artefacto generado localmente por src/train.py."""
    return joblib.load(MODEL_PATH)


def predict(text):
    model = load_model()
    label = int(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    scores = {
        CLASS_NAMES[int(code)]: float(probability)
        for code, probability in zip(model.classes_, probabilities)
    }
    return {"predicted_class": CLASS_NAMES[label], "probabilities": scores}
