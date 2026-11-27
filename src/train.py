import csv
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from codecarbon import OfflineEmissionsTracker
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
CARBON_DIR = EXPERIMENTS / "carbon"
MODELS = ROOT / "models"

CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")
LABELS = tuple(range(len(CLASS_NAMES)))

DATASET_ID = "fancyzhx/ag_news"
VALIDATION_FRACTION = 0.2


CONFIGURATIONS = (
    {
        "id": "logistic_unigrams",
        "model": "LogisticRegression",
        "ngram_range": (1, 1),
        "C": 1.0,
    },
    {
        "id": "logistic_bigrams",
        "model": "LogisticRegression",
        "ngram_range": (1, 2),
        "C": 1.0,
    },
    {
        "id": "complement_nb_bigrams",
        "model": "ComplementNB",
        "ngram_range": (1, 2),
        "alpha": 1.0,
    },
    {
        "id": "sgd_bigrams",
        "model": "SGDClassifier",
        "ngram_range": (1, 2),
        "loss": "log_loss",
        "alpha": 1e-4,
    },
)


def build_pipeline(config, seed=RANDOM_STATE):
    """Construye un pipeline TF-IDF + clasificador."""

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=config["ngram_range"],
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    if config["model"] == "LogisticRegression":
        classifier = LogisticRegression(
            C=config["C"],
            max_iter=1000,
            random_state=seed,
        )

    elif config["model"] == "ComplementNB":
        classifier = ComplementNB(
            alpha=config["alpha"],
        )

    elif config["model"] == "SGDClassifier":
        classifier = SGDClassifier(
            loss=config["loss"],
            alpha=config["alpha"],
            max_iter=1000,
            random_state=seed,
        )

    else:
        raise ValueError(
            f"Modelo desconocido: {config['model']}"
        )

    return Pipeline(
        [
            ("tfidf", tfidf),
            ("classifier", classifier),
        ]
    )


def measure(model, x, y):
    """Calcula métricas y tiempo de predicción."""

    started = perf_counter()
    predictions = model.predict(x)
    elapsed = perf_counter() - started

    report = classification_report(
        y,
        predictions,
        labels=LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "f1_macro": f1_score(
            y,
            predictions,
            labels=LABELS,
            average="macro",
        ),
        "predict_seconds": elapsed,
        "per_class": {
            name: {
                key: report[name][key]
                for key in (
                    "precision",
                    "recall",
                    "f1-score",
                    "support",
                )
            }
            for name in CLASS_NAMES
        },
    }

    return metrics, predictions


def read_carbon_file(carbon_path):
    """Lee el último registro generado por CodeCarbon."""

    values = {
        "energy_kwh": None,
        "emissions_kg_co2eq": None,
        "cpu_energy_kwh": None,
        "ram_energy_kwh": None,
        "gpu_energy_kwh": None,
    }

    if not carbon_path.exists():
        return values

    with carbon_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return values

    last_row = rows[-1]

    columns = {
        "energy_kwh": "energy_consumed",
        "emissions_kg_co2eq": "emissions",
        "cpu_energy_kwh": "cpu_energy",
        "ram_energy_kwh": "ram_energy",
        "gpu_energy_kwh": "gpu_energy",
    }

    for result_key, csv_key in columns.items():
        value = last_row.get(csv_key)

        if value not in (None, ""):
            values[result_key] = float(value)

    return values


def train_and_measure(
    config,
    x_fit,
    y_fit,
    x_val,
    y_val,
):
    """Entrena y evalúa una configuración con medición ambiental."""

    CARBON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    carbon_file = f"{config['id']}_validation.csv"
    carbon_path = CARBON_DIR / carbon_file

    tracker = OfflineEmissionsTracker(
        country_iso_code="COL",
        tracking_mode="process",
        measure_power_secs=1,
        output_dir=str(CARBON_DIR),
        output_file=carbon_file,
        project_name=(
            f"NewsScope-{config['id']}-validation"
        ),
    )

    pipeline = build_pipeline(config)

    tracker.start()

    try:
        started = perf_counter()

        pipeline.fit(x_fit, y_fit)

        train_seconds = perf_counter() - started

        metrics, predictions = measure(
            pipeline,
            x_val,
            y_val,
        )

    finally:
        tracker.stop()

    carbon_values = read_carbon_file(carbon_path)

    carbon_values["carbon_file"] = str(
        carbon_path.relative_to(ROOT)
    )

    carbon_values["feature_count"] = len(
        pipeline.named_steps["tfidf"].vocabulary_
    )

    return (
        pipeline,
        metrics,
        train_seconds,
        carbon_values,
        predictions,
    )


def flatten_row(
    config,
    metrics,
    train_seconds,
    train_count,
    evaluation_count,
    seed,
):
    """Convierte las métricas a una fila del CSV comparativo."""

    row = {
        "id": config["id"],
        "model": config["model"],
        "vectorizer": "TF-IDF",
        "ngram_range": str(config["ngram_range"]),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True,
        "hyperparameters": json.dumps(
            {
                key: value
                for key, value in config.items()
                if key not in (
                    "id",
                    "model",
                    "ngram_range",
                )
            },
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
    """Escribe una lista de diccionarios en un CSV."""

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_figures(y_true, predictions, metrics):
    """Guarda matriz de confusión y F1 por clase."""

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=LABELS,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(
        LABELS,
        CLASS_NAMES,
        rotation=30,
        ha="right",
    )
    ax.set_yticks(LABELS, CLASS_NAMES)
    ax.set(
        xlabel="Predicción",
        ylabel="Clase real",
        title="Matriz de confusión (prueba)",
    )

    for row in LABELS:
        for col in LABELS:
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(
        EXPERIMENTS / "confusion_matrix.png",
        dpi=150,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        CLASS_NAMES,
        [
            metrics["per_class"][name]["f1-score"]
            for name in CLASS_NAMES
        ],
    )
    ax.set(
        ylim=(0, 1),
        ylabel="F1",
        title="F1 por clase (prueba)",
    )
    fig.tight_layout()
    fig.savefig(
        EXPERIMENTS / "f1_per_class.png",
        dpi=150,
    )
    plt.close(fig)

    return matrix.tolist()


def main():
    EXPERIMENTS.mkdir(
        parents=True,
        exist_ok=True,
    )
    CARBON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    MODELS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Cargando {DATASET_ID}...",
        flush=True,
    )

    dataset = load_dataset(DATASET_ID)

    x_train = dataset["train"]["text"]
    y_train = dataset["train"]["label"]

    x_test = dataset["test"]["text"]
    y_test = dataset["test"]["label"]

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_FRACTION,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    rows = []

    for config in CONFIGURATIONS:
        print(
            f"Validación: {config['id']}",
            flush=True,
        )

        (
            pipeline,
            metrics,
            train_seconds,
            carbon_values,
            predictions,
        ) = train_and_measure(
            config,
            x_fit,
            y_fit,
            x_val,
            y_val,
        )

        row = flatten_row(
            config,
            metrics,
            train_seconds,
            len(x_fit),
            len(x_val),
            RANDOM_STATE,
        )

        row["energy_kwh"] = carbon_values["energy_kwh"]
        row["emissions_kg_co2eq"] = carbon_values[
            "emissions_kg_co2eq"
        ]
        row["cpu_energy_kwh"] = carbon_values[
            "cpu_energy_kwh"
        ]
        row["ram_energy_kwh"] = carbon_values[
            "ram_energy_kwh"
        ]
        row["gpu_energy_kwh"] = carbon_values[
            "gpu_energy_kwh"
        ]
        row["feature_count"] = carbon_values[
            "feature_count"
        ]
        row["carbon_measurement_file"] = carbon_values[
            "carbon_file"
        ]

        rows.append(row)

    winner = max(
        range(len(rows)),
        key=lambda index: rows[index]["f1_macro"],
    )

    best = CONFIGURATIONS[winner]

    write_csv(
        EXPERIMENTS / "model_comparison.csv",
        rows,
    )

    print(
        f"Seleccionado en validación: {best['id']}",
        flush=True,
    )

    # Comprobación de estabilidad con una segunda semilla.
    repeat_seed = RANDOM_STATE + 1
    repeated = build_pipeline(
        best,
        repeat_seed,
    )

    started = perf_counter()
    repeated.fit(x_fit, y_fit)
    repeat_train_seconds = perf_counter() - started

    repeat_metrics, _ = measure(
        repeated,
        x_val,
        y_val,
    )

    write_csv(
        EXPERIMENTS / "stability.csv",
        [
            rows[winner],
            flatten_row(
                best,
                repeat_metrics,
                repeat_train_seconds,
                len(x_fit),
                len(x_val),
                repeat_seed,
            ),
        ],
    )

    del repeated

    # Entrenamiento final del ganador y evaluación única en prueba.
    final_carbon_file = "final_model.csv"
    final_carbon_path = CARBON_DIR / final_carbon_file

    final_tracker = OfflineEmissionsTracker(
        country_iso_code="COL",
        tracking_mode="process",
        measure_power_secs=1,
        output_dir=str(CARBON_DIR),
        output_file=final_carbon_file,
        project_name="NewsScope-final-model",
    )

    final_model = build_pipeline(best)

    final_tracker.start()

    try:
        started = perf_counter()

        final_model.fit(x_train, y_train)

        final_train_seconds = perf_counter() - started

        test_metrics, predictions = measure(
            final_model,
            x_test,
            y_test,
        )

    finally:
        final_tracker.stop()

    final_carbon_values = read_carbon_file(
        final_carbon_path
    )

    matrix = save_figures(
        y_test,
        predictions,
        test_metrics,
    )

    model_path = MODELS / "newsscope.joblib"
    joblib.dump(final_model, model_path)

    model_size_mb = model_path.stat().st_size / (
        1024 ** 2
    )

    final_feature_count = len(
        final_model.named_steps["tfidf"].vocabulary_
    )

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
        "repeat_validation_f1_macro": (
            repeat_metrics["f1_macro"]
        ),
        "final_train_seconds": final_train_seconds,
        "test": test_metrics,
        "test_confusion_matrix": matrix,
        "model_path": "models/newsscope.joblib",
        "final_computational_cost": {
            "train_seconds": final_train_seconds,
            "predict_seconds": test_metrics[
                "predict_seconds"
            ],
            "energy_kwh": final_carbon_values[
                "energy_kwh"
            ],
            "emissions_kg_co2eq": (
                final_carbon_values[
                    "emissions_kg_co2eq"
                ]
            ),
            "cpu_energy_kwh": final_carbon_values[
                "cpu_energy_kwh"
            ],
            "ram_energy_kwh": final_carbon_values[
                "ram_energy_kwh"
            ],
            "gpu_energy_kwh": final_carbon_values[
                "gpu_energy_kwh"
            ],
            "carbon_file": str(
                final_carbon_path.relative_to(ROOT)
            ),
            "feature_count": final_feature_count,
            "model_size_mb": model_size_mb,
        },
    }

    with (
        EXPERIMENTS / "run_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            summary,
            output,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Prueba final: "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"F1 macro={test_metrics['f1_macro']:.4f}",
        flush=True,
    )

    print(
        "Costo computacional final: "
        f"{final_train_seconds:.2f} segundos",
        flush=True,
    )

    print(
        "Emisiones finales: "
        f"{final_carbon_values['emissions_kg_co2eq']}",
        flush=True,
    )


if __name__ == "__main__":
    main()