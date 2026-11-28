from pathlib import Path
import json

from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from config import RANDOM_STATE


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando AG News...")
    dataset = load_dataset("fancyzhx/ag_news")

    x_train = dataset["train"]["text"]
    y_train = dataset["train"]["label"]

    x_test = dataset["test"]["text"]
    y_test = dataset["test"]["label"]

    print(f"Ejemplos de entrenamiento: {len(x_train)}")
    print(f"Ejemplos de prueba: {len(x_test)}")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 1),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ])

    print("Entrenando modelo...")
    model.fit(x_train, y_train)

    print("Generando predicciones...")
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    class_names = [
        "World",
        "Sports",
        "Business",
        "Sci/Tech"
    ]

    report = classification_report(
        y_test,
        predictions,
        target_names=class_names
    )

    results = {
        "dataset": "AG News",
        "model": "LogisticRegression",
        "vectorizer": "TF-IDF",
        "ngram_range": [1, 1],
        "train_examples": len(x_train),
        "test_examples": len(x_test),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
        "status": "first baseline result"
    }

    json_path = results_dir / "baseline_results.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    report_path = results_dir / "baseline_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("NewsScope - Resultado de línea base\n")
        file.write("=" * 40 + "\n\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"F1 macro: {f1_macro:.4f}\n\n")
        file.write("Reporte por clase:\n")
        file.write(report)

    print("\nResultados de la línea base")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print("\nArchivos guardados:")
    print(f"- {json_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
