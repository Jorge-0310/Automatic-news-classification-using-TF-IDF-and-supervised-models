"""Proyección pública de experimentos; nunca lee el CSV local con noticias."""

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")


def read_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def dashboard_metrics():
    """Construye métricas a partir de la ejecución registrada, sin corpus."""
    run = read_json(EXPERIMENTS / "run_config.json")
    comparison = read_csv(EXPERIMENTS / "model_comparison.csv")
    stability = read_csv(EXPERIMENTS / "stability.csv")
    config_ids = [entry["id"] for entry in run["configurations"]]
    if [row["id"] for row in comparison] != config_ids:
        raise ValueError("Comparación y configuración no corresponden a la misma ejecución")
    if run["selected_id"] not in config_ids:
        raise ValueError("El ganador no aparece en la comparación")
    if len(run["test_confusion_matrix"]) != len(CLASS_NAMES) or any(
        len(row) != len(CLASS_NAMES) for row in run["test_confusion_matrix"]
    ):
        raise ValueError("Matriz de confusión inválida")
    selected = next(row for row in comparison if row["id"] == run["selected_id"])
    if abs(float(selected["f1_macro"]) - run["validation_f1_macro"]) > 1e-8:
        raise ValueError("F1 de validación incoherente")
    if any(row["id"] != run["selected_id"] for row in stability):
        raise ValueError("Archivo de estabilidad incoherente")

    # Proyección explícita: no filtrar accidentalmente rutas, notas o textos.
    return {
        "dataset": run["dataset"],
        "train_count": run["fit_count"] + run["validation_count"],
        "validation_count": run["validation_count"],
        "test_count": run["test_count"],
        "seed": run["random_state"],
        "selection_metric": run["selection_metric"],
        "selected_by": run["selected_by"],
        "selected_id": run["selected_id"],
        "classes": list(CLASS_NAMES),
        "test": {
            "accuracy": run["test"]["accuracy"],
            "f1_macro": run["test"]["f1_macro"],
            "per_class": run["test"]["per_class"],
        },
        "confusion_matrix": run["test_confusion_matrix"],
        "comparison": [
            {
                "id": row["id"],
                "model": row["model"],
                "ngram_range": row["ngram_range"],
                "accuracy": float(row["accuracy"]),
                "f1_macro": float(row["f1_macro"]),
                "train_seconds": float(row["train_seconds"]),
            }
            for row in comparison
        ],
        "stability": [
            {"seed": int(row["random_state"]), "f1_macro": float(row["f1_macro"])}
            for row in stability
        ],
        "methodology": (
            "Selección por F1 macro en validación estratificada; "
            "solo el modelo ganador se reentrenó y evaluó en prueba."
        ),
    }


def error_summary():
    """Solo agregados; ignora deliberadamente texto y notas libres de casos."""
    path = EXPERIMENTS / "error_review_confirmed.json"
    if not path.is_file():
        return {"status": "pending", "count": 0, "types": [], "pairs": []}
    review = read_json(path)
    cases = review.get("cases", [])
    if review.get("count") != len(cases) or len(cases) < 10:
        raise ValueError("Revisión de errores incompleta o inconsistente")
    pairs = Counter(
        (case["true_class"], case["predicted_class"]) for case in cases
        if case["true_class"] in CLASS_NAMES and case["predicted_class"] in CLASS_NAMES
    )
    if sum(pairs.values()) != len(cases):
        raise ValueError("Clases inválidas en revisión de errores")
    # "Tipo" público = par de confusión. No exponer error_type ni hipótesis
    # redactadas libremente: pueden incluir fragmentos del corpus protegido.
    public_pairs = [
        {"true_class": true, "predicted_class": predicted, "count": count}
        for (true, predicted), count in sorted(
            pairs.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "status": "confirmed",
        "count": len(cases),
        "types": [
            {"name": f"{item['true_class']} → {item['predicted_class']}",
             "count": item["count"]}
            for item in public_pairs
        ],
        "pairs": public_pairs,
        "limitations": [
            "Una sola etiqueta puede ocultar temas simultáneos.",
            "Los textos breves pueden carecer de contexto suficiente.",
            "Las hipótesis de causa no demuestran cómo decidió el modelo.",
        ],
        "note": "Recuentos de revisión declarada por el equipo; no incluyen noticias.",
    }


if __name__ == "__main__":
    output = EXPERIMENTS / "dashboard_metrics.json"
    with output.open("w", encoding="utf-8") as destination:
        json.dump(dashboard_metrics(), destination, indent=2, ensure_ascii=False)
    print(f"Métricas públicas actualizadas: {output}")
    summary_path = EXPERIMENTS / "error_summary_public.json"
    with summary_path.open("w", encoding="utf-8") as destination:
        json.dump(error_summary(), destination, indent=2, ensure_ascii=False)
    print(f"Resumen público sin textos: {summary_path}")
