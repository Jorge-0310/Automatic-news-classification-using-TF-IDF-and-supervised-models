"""Prueba de integración con el modelo local; ejecutar tras python src/train.py."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from api import app
from inference import MODEL_PATH, load_model


EXAMPLES = {
    "World": "World leaders signed a new international treaty after diplomatic talks.",
    "Sports": "The football team won the championship match after scoring three goals.",
    "Business": "The company reported higher quarterly profits and rising stock prices.",
    "Sci/Tech": "Scientists launched a spacecraft to explore distant planets with a telescope.",
}


def main():
    if not MODEL_PATH.is_file():
        raise SystemExit("Falta models/newsscope.joblib; ejecute python src/train.py")
    client = TestClient(app)
    for route in ("/", "/static/styles.css", "/static/app.js",
                  "/api/health", "/api/model", "/api/metrics", "/api/errors/summary"):
        response = client.get(route)
        assert response.status_code == 200, (route, response.status_code)
    for name, text in EXAMPLES.items():
        result = client.post("/api/predict", json={"text": text})
        assert result.status_code == 200, (name, result.status_code)
        data = result.json()
        assert data["predicted_class"] == name, (name, data["predicted_class"])
        assert len(data["probabilities"]) == 4
        assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-6
        assert text not in result.text
    assert client.post("/api/predict", json={"text": "  "}).status_code == 422
    assert client.post("/api/predict", json={"text": "x" * 5001}).status_code == 413
    assert load_model.cache_info().currsize == 1  # No deserializar por petición.
    print("Frontend/API: cuatro categorías, métricas, errores seguros y controles verificados.")


if __name__ == "__main__":
    main()
