"""API local NewsScope. Iniciar: uvicorn api:app --no-access-log."""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dashboard_data import dashboard_metrics, error_summary  # noqa: E402
from inference import CLASS_NAMES, MODEL_PATH, load_model, predict  # noqa: E402


MAX_TEXT_LENGTH = 5000
WARNING = (
    "Clasificación temática experimental; probabilidades no calibradas. "
    "No verifica veracidad ni reemplaza revisión humana."
)
app = FastAPI(title="NewsScope", version="1.0.0", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


class PredictRequest(BaseModel):
    text: str


@app.exception_handler(RequestValidationError)
async def validation_error(_request, _error):
    # FastAPI por defecto puede incluir `input` en errores de validación.
    return JSONResponse(status_code=422, content={"detail": "Solicitud JSON inválida."})


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/health")
def health():
    try:
        metrics = dashboard_metrics()
        name = metrics["selected_id"]
    except (OSError, ValueError, KeyError):
        name = None
    return {
        "status": "ok",
        "model_loaded": bool(load_model.cache_info().currsize),
        "model_available": MODEL_PATH.is_file(),
        "model_name": name,
        "version": app.version,
    }


@app.get("/api/model")
def model_info():
    try:
        metrics = dashboard_metrics()
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="Métricas no disponibles.") from exc
    selected = next(row for row in metrics["comparison"] if row["id"] == metrics["selected_id"])
    return {
        "name": metrics["selected_id"],
        "classes": list(CLASS_NAMES),
        "dataset": metrics["dataset"],
        "random_state": metrics["seed"],
        "vectorizer": "TF-IDF",
        "ngram_range": selected["ngram_range"],
        "validation_count": metrics["validation_count"],
        "test_count": metrics["test_count"],
        "model_available": MODEL_PATH.is_file(),
        "version": app.version,
        "warning": WARNING,
    }


@app.post("/api/predict")
def prediction(payload: PredictRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Introduzca una noticia en inglés.")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413, detail=f"El texto debe tener máximo {MAX_TEXT_LENGTH} caracteres."
        )
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503, detail="Modelo no disponible: ejecute python src/train.py."
        )
    try:
        result = predict(text)
    except (OSError, ValueError, AttributeError) as exc:
        raise HTTPException(status_code=503, detail="No se pudo cargar el modelo local.") from exc
    try:
        name = dashboard_metrics()["selected_id"]
    except (OSError, ValueError, KeyError):
        name = "modelo local"
    return {**result, "model": name, "warning": WARNING}


@app.get("/api/metrics")
def metrics():
    try:
        return dashboard_metrics()
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="Métricas no disponibles.") from exc


@app.get("/api/errors/summary")
def errors():
    try:
        return error_summary()
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="Resumen de errores no disponible.") from exc
