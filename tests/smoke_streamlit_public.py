"""Prueba de la app pública con el modelo local ignorado por Git.

python tests/smoke_streamlit_public.py
"""

import hashlib
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "newsscope.joblib"
SCRIPT = ROOT / "deploy" / "streamlit_app.py"
EXPECTED_HASH = (ROOT / "deploy" / "model.sha256").read_text(encoding="ascii").strip()
EXAMPLES = (
    ("World", "World leaders signed a new international treaty after diplomatic talks."),
    ("Sports", "The football team won the championship match after scoring three goals."),
    ("Business", "The company reported higher quarterly profits and rising stock prices."),
    ("Sci/Tech", "Scientists launched a spacecraft to explore distant planets with a telescope."),
)


def main():
    if not MODEL.is_file():
        raise SystemExit("Falta models/newsscope.joblib; ejecute python src/train.py")
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == EXPECTED_HASH
    app = AppTest.from_file(str(SCRIPT), default_timeout=90).run()
    assert not app.exception, [error.message for error in app.exception]
    assert app.radio[0].value == "01   Clasificar"
    submit = lambda: next(button for button in app.button if button.label.startswith("Clasificar noticia"))
    next(button for button in app.button if button.label == "Sports").click().run()
    assert app.text_area[0].value.startswith("The football team")
    app.text_area[0].set_value("")
    submit().click().run()
    assert any("Escribe primero" in warning.value for warning in app.warning)
    for expected, text in EXAMPLES:
        app.text_area[0].set_value(text)
        submit().click().run()
        assert not app.exception, [error.message for error in app.exception]
        assert any(f"## {expected}" in item.value for item in app.markdown), expected
        assert len(app.get("progress")) == 4
    submit().click().run()  # Quinta solicitud.
    assert any("## Sci/Tech" in item.value for item in app.markdown)
    submit().click().run()  # Sexta solicitud: bloqueada.
    assert any("Límite temporal" in warning.value for warning in app.warning)
    app.radio[0].set_value("02   Rendimiento").run()
    assert not app.exception and len(app.metric) == 4
    app.radio[0].set_value("03   Errores").run()
    summary = json.loads(
        (ROOT / "experiments" / "error_summary_public.json").read_text(encoding="utf-8")
    )
    assert not app.exception and app.metric[0].value == str(summary["count"])
    app.radio[0].set_value("04   El modelo").run()
    assert not app.exception
    metrics = json.loads(
        (ROOT / "experiments" / "dashboard_metrics.json").read_text(encoding="utf-8")
    )
    cost = metrics["final_computational_cost"]
    assert app.metric[0].value == f"{cost['train_seconds']:.1f} s"
    assert app.metric[1].value == f"{cost['energy_kwh'] * 1000:.2f} Wh"
    assert app.metric[2].value == f"{cost['emissions_kg_co2eq'] * 1000:.3f} g CO₂e"
    assert app.metric[3].value == f"{cost['model_size_mb']:.1f} MiB"
    print("Streamlit público: hash, 4 predicciones, límite y 3 vistas verificados.")


if __name__ == "__main__":
    main()
