"""Revisión HUMANA interactiva de errores; no publica textos del corpus.

Ejecutar después de src/error_analysis.py y opcionalmente annotate_errors.py.
El programa no puede verificar por sí mismo que el usuario leyó el texto.
"""

import csv
import json
from collections import Counter
from datetime import date

from train import EXPERIMENTS


def request(label):
    return input(f"{label}: ").strip()


def main():
    source = EXPERIMENTS / "errors.csv"
    target = EXPERIMENTS / "error_review_confirmed.json"
    if not source.exists():
        raise SystemExit("Falta errors.csv: ejecute primero python src/error_analysis.py")
    if target.exists():
        raise SystemExit(f"{target} ya existe; no se sobrescriben revisiones previas")
    reviewer = request("Nombre del integrante que revisa")
    if not reviewer:
        raise SystemExit("Se requiere identificar al integrante revisor.")
    with source.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    checked = []
    for row in rows:
        print("\n" + "=" * 60)
        print("Índice:", row["test_index"], "| Real:", row["true_class"],
              "| Predicha:", row["predicted_class"])
        print("Texto:", row["text"])
        print("Probabilidad predicha:", row["predicted_probability"])
        print("Borrador IA:", row["error_type"] or "(sin propuesta)")
        decision = request("¿Revisó el texto? [a=aceptar borrador, e=editar, s=saltar]").lower()
        if decision == "s":
            continue
        if decision not in ("a", "e"):
            print("Opción no válida; caso omitido.")
            continue
        keys = ("error_type", "cause_hypothesis", "limitation")
        if decision == "a" and not all(row[key].strip() for key in keys):
            print("No hay borrador completo; use 'e' para redactar su revisión.")
            continue
        if decision == "a":
            category, hypothesis, limitation = (row[key] for key in keys)
        else:
            category = request("Tipo de error")
            hypothesis = request("Hipótesis de causa (no hecho probado)")
            limitation = request("Limitación que ilustra")
            if not all((category, hypothesis, limitation)):
                print("Revisión incompleta; caso omitido.")
                continue
        checked.append({
            "test_index": int(row["test_index"]),
            "true_class": row["true_class"],
            "predicted_class": row["predicted_class"],
            "error_type": category,
            "cause_hypothesis": hypothesis,
            "limitation": limitation,
        })

    if len(checked) < 10:
        print(f"Solo {len(checked)} revisados; se necesitan >=10. No se guardó confirmación.")
        return
    summary = Counter(item["error_type"] for item in checked)
    document = {
        "reviewer": reviewer,
        "review_date": date.today().isoformat(),
        "count": len(checked),
        "by_error_type": dict(sorted(summary.items())),
        "cases": checked,
        "note": "Revisión declarada por integrante; hipótesis no prueban causalidad.",
    }
    with target.open("x", encoding="utf-8") as output:
        json.dump(document, output, indent=2, ensure_ascii=False)
    print(f"{len(checked)} revisiones en {target}; sin textos de noticias.")


if __name__ == "__main__":
    main()
