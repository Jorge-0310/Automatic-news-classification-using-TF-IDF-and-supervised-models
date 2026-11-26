"""Comparación reproducible de TF-IDF + tres clasificadores sobre AG News.

Selección exclusivamente en validación estratificada; prueba solo para el
pipeline elegido, reentrenado con los 120 000 ejemplos de entrenamiento.
"""

import csv
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB

from config import RANDOM_STATE


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
MODELS = ROOT / "models"
CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")
LABELS = tuple(range(len(CLASS_NAMES)))
DATASET_ID = "fancyzhx/ag_news"
VALIDATION_FRACTION = 0.2

# Conjunto pequeño y explícito de configuraciones, sin búsqueda en test.
# La primera reproduce los hiperparámetros de la línea base original.
CONFIGURATIONS = (
    {"id": "logistic_unigrams", "model": "LogisticRegression", "ngram_range": (1, 1), "C": 1.0},
    {"id": "logistic_bigrams", "model": "LogisticRegression", "ngram_range": (1, 2), "C": 1.0},
    {"id": "complement_nb_bigrams", "model": "ComplementNB", "ngram_range": (1, 2), "alpha": 1.0},
    {"id": "sgd_bigrams", "model": "SGDClassifier", "ngram_range": (1, 2), "loss": "log_loss", "alpha": 1e-4},
)


def build_pipeline(config, seed=RANDOM_STATE):
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=config["ngram_range"],
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    if config["model"] == "LogisticRegression":
        classifier = LogisticRegression(C=config["C"], max_iter=1000, random_state=seed)
    elif config["model"] == "ComplementNB":
        classifier = ComplementNB(alpha=config["alpha"])
    elif config["model"] == "SGDClassifier":
        classifier = SGDClassifier(
            loss=config["loss"], alpha=config["alpha"], max_iter=1000, random_state=seed
        )
    else:
        raise ValueError(f"Modelo desconocido: {config['model']}")
    return Pipeline([("tfidf", tfidf), ("classifier", classifier)])


def measure(model, x, y):
    started = perf_counter()
    predictions = model.predict(x)
    elapsed = perf_counter() - started
    report = classification_report(
        y, predictions, labels=LABELS, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y, predictions),
        "f1_macro": f1_score(y, predictions, labels=LABELS, average="macro"),
        "predict_seconds": elapsed,
        "per_class": {
            name: {key: report[name][key] for key in ("precision", "recall", "f1-score", "support")}
            for name in CLASS_NAMES
        },
    }, predictions


def flatten_row(config, metrics, train_seconds, train_count, evaluation_count, seed):
    row = {
        "id": config["id"],
        "model": config["model"],
        "vectorizer": "TF-IDF",
        "ngram_range": str(config["ngram_range"]),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True,
        "hyperparameters": json.dumps(
            {k: v for k, v in config.items() if k not in ("id", "model", "ngram_range")},
            sort_keys=True,
        ),
        "random_state": seed,
        "train_count": train_count,
        "evaluation_count": evaluation_count,
        "train_seconds": train_seconds,
        "predict_seconds": metrics["predict_seconds"],
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
    }
    for name in CLASS_NAMES:
        for metric, value in metrics["per_class"][name].items():
            row[f"{name}_{metric}"] = value
    return row


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_figures(y_true, predictions, metrics):
    matrix = confusion_matrix(y_true, predictions, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(LABELS, CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(LABELS, CLASS_NAMES)
    ax.set(xlabel="Predicción", ylabel="Clase real", title="Matriz de confusión (prueba)")
    for row in LABELS:
        for col in LABELS:
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(EXPERIMENTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(CLASS_NAMES, [metrics["per_class"][name]["f1-score"] for name in CLASS_NAMES])
    ax.set(ylim=(0, 1), ylabel="F1", title="F1 por clase (prueba)")
    fig.tight_layout()
    fig.savefig(EXPERIMENTS / "f1_per_class.png", dpi=150)
    plt.close(fig)
    return matrix.tolist()


def main():
    EXPERIMENTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    print(f"Cargando {DATASET_ID}...", flush=True)
    dataset = load_dataset(DATASET_ID)
    x_train = dataset["train"]["text"]
    y_train = dataset["train"]["label"]
    x_test = dataset["test"]["text"]
    y_test = dataset["test"]["label"]

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=VALIDATION_FRACTION,
        stratify=y_train, random_state=RANDOM_STATE,
    )
    rows = []
    for config in CONFIGURATIONS:
        print(f"Validación: {config['id']}", flush=True)
        pipeline = build_pipeline(config)
        started = perf_counter()
        pipeline.fit(x_fit, y_fit)
        train_seconds = perf_counter() - started
        metrics, _ = measure(pipeline, x_val, y_val)
        rows.append(flatten_row(config, metrics, train_seconds, len(x_fit), len(x_val), RANDOM_STATE))

    # El orden en CONFIGURATIONS resuelve empates de forma determinista.
    winner = max(range(len(rows)), key=lambda index: rows[index]["f1_macro"])
    best = CONFIGURATIONS[winner]
    write_csv(EXPERIMENTS / "model_comparison.csv", rows)
    print(f"Seleccionado en validación: {best['id']}", flush=True)

    # Comprobación de estabilidad: repetir la configuración ganadora con otra
    # semilla, sobre EXACTAMENTE la misma partición; no decide el ganador.
    repeat_seed = RANDOM_STATE + 1
    repeated = build_pipeline(best, repeat_seed)
    started = perf_counter()
    repeated.fit(x_fit, y_fit)
    repeat_train_seconds = perf_counter() - started
    repeat_metrics, _ = measure(repeated, x_val, y_val)
    write_csv(
        EXPERIMENTS / "stability.csv",
        [rows[winner], flatten_row(best, repeat_metrics, repeat_train_seconds,
                                    len(x_fit), len(x_val), repeat_seed)],
    )
    del repeated

    final_model = build_pipeline(best)
    started = perf_counter()
    final_model.fit(x_train, y_train)
    final_train_seconds = perf_counter() - started
    test_metrics, predictions = measure(final_model, x_test, y_test)
    matrix = save_figures(y_test, predictions, test_metrics)
    joblib.dump(final_model, MODELS / "newsscope.joblib")

    summary = {
        "dataset": DATASET_ID,
        "random_state": RANDOM_STATE,
        "validation_fraction": VALIDATION_FRACTION,
        "selection_metric": "f1_macro",
        "validation_count": len(x_val),
        "fit_count": len(x_fit),
        "test_count": len(x_test),
        "configurations": CONFIGURATIONS,
        "selected_id": best["id"],
        "selected_by": "validation_only",
        "validation_f1_macro": rows[winner]["f1_macro"],
        "repeat_seed": repeat_seed,
        "repeat_validation_f1_macro": repeat_metrics["f1_macro"],
        "final_train_seconds": final_train_seconds,
        "test": test_metrics,
        "test_confusion_matrix": matrix,
        "model_path": "models/newsscope.joblib",
    }
    with (EXPERIMENTS / "run_config.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
    print(f"Prueba final: accuracy={test_metrics['accuracy']:.4f}, "
          f"F1 macro={test_metrics['f1_macro']:.4f}", flush=True)


if __name__ == "__main__":
    main()
