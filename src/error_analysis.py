"""Extrae errores reales de prueba para anotación MANUAL, sin publicarlos.

Ejecutar después de train.py. El CSV contiene texto del corpus y .gitignore
impide versionarlo; no se asignan automáticamente causas a ejemplos.
"""

import argparse
import csv
from collections import Counter

import joblib
from datasets import load_dataset

from train import CLASS_NAMES, DATASET_ID, EXPERIMENTS, MODELS


def select_error_indices(true_labels, predictions, count):
    errors = [
        i for i, (true, predicted) in enumerate(zip(true_labels, predictions))
        if true != predicted
    ]
    # Recorrer los pares más confundidos por turnos para ofrecer ejemplos de
    # varios tipos de confusión. Orden estable por índice dentro de cada par.
    pairs = Counter((true_labels[i], int(predictions[i])) for i in errors)
    ordered_pairs = [pair for pair, _ in sorted(pairs.items(), key=lambda item: (-item[1], item[0]))]
    buckets = {
        pair: iter([
            i for i in errors if (true_labels[i], int(predictions[i])) == pair
        ])
        for pair in ordered_pairs
    }
    selected = []
    while len(selected) < count:
        added = False
        for pair in ordered_pairs:
            candidate = next(buckets[pair], None)
            if candidate is not None:
                selected.append(candidate)
                added = True
            if len(selected) == count:
                break
        if not added:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description="Extraer errores para revisión manual")
    parser.add_argument("--count", type=int, default=20, help="Candidatos a revisar (mínimo 10)")
    args = parser.parse_args()
    if args.count < 10:
        parser.error("--count debe ser al menos 10")

    path = EXPERIMENTS / "errors.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as existing:
            previous = list(csv.DictReader(existing))
        if any(row.get(key, "").strip() for row in previous
               for key in ("error_type", "cause_hypothesis", "limitation")):
            parser.error(f"{path} contiene anotaciones; se preserva sin sobrescribir")
    model_path = MODELS / "newsscope.joblib"
    if not model_path.exists():
        parser.error("Falta models/newsscope.joblib; ejecute primero python src/train.py")

    model = joblib.load(model_path)  # Cargar solo modelos locales de confianza.
    dataset = load_dataset(DATASET_ID)["test"]
    predictions = model.predict(dataset["text"])
    probabilities = model.predict_proba(dataset["text"])
    classes = list(model.classes_)
    selected = select_error_indices(dataset["label"], predictions, args.count)
    EXPERIMENTS.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=(
            "test_index", "text", "true_class", "predicted_class",
            "predicted_probability", "error_type", "cause_hypothesis", "limitation",
        ))
        writer.writeheader()
        for i in selected:
            predicted = int(predictions[i])
            writer.writerow({
                "test_index": i,
                "text": dataset["text"][i],
                "true_class": CLASS_NAMES[dataset["label"][i]],
                "predicted_class": CLASS_NAMES[predicted],
                "predicted_probability": float(probabilities[i][classes.index(predicted)]),
                "error_type": "",
                "cause_hypothesis": "",
                "limitation": "",
            })
    print(f"{len(selected)} errores reales en {path}. "
          "Complete manualmente tipo, hipótesis y limitación; no publique el CSV.")


if __name__ == "__main__":
    main()
