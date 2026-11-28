"""Aplica borradores de revisión a errores locales; no sustituye revisión humana."""

import csv
import json

from train import EXPERIMENTS


def main():
    path = EXPERIMENTS / "errors.csv"
    if not path.exists():
        raise SystemExit("Ejecute primero python src/error_analysis.py")
    with (EXPERIMENTS / "error_review_draft.json").open(encoding="utf-8") as source:
        annotations = json.load(source)["annotations"]
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = list(reader)

    if len({row["test_index"] for row in rows}) != len(rows):
        raise SystemExit("El CSV tiene índices duplicados; regenérelo antes de anotar")
    available = {row["test_index"] for row in rows}
    if not annotations.keys() <= available:
        raise SystemExit(f"Faltan índices: {set(annotations) - available}")
    for row in rows:
        annotation = annotations.get(row["test_index"])
        if annotation:
            if (row["true_class"], row["predicted_class"]) != (
                annotation["true_class"], annotation["predicted_class"]
            ):
                raise SystemExit(f"Clases distintas para índice {row['test_index']}")
            for field in ("error_type", "cause_hypothesis", "limitation"):
                if row[field].strip() and row[field] != annotation[field]:
                    raise SystemExit(f"Anotación previa distinta en {row['test_index']}: {field}")
                row[field] = annotation[field]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Borradores aplicados a {len(annotations)} errores locales. "
          "Revisar cada hipótesis y clase con el equipo antes de reportarlas.")


if __name__ == "__main__":
    main()
