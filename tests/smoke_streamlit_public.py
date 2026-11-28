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
    app.button[4].click().run()
    assert any("Escribe primero" in warning.value for warning in app.warning)
    for expected, text in EXAMPLES:
        app.text_area[0].set_value(text)
        app.button[4].click().run()
        assert not app.exception, [error.message for error in app.exception]
        assert any(expected in message.value for message in app.success), expected
    app.button[4].click().run()  # Quinta solicitud.
    assert any(message.value.startswith("Categoría predicha") for message in app.success)
    app.button[4].click().run()  # Sexta solicitud: bloqueada.
    assert any("Límite temporal" in warning.value for warning in app.warning)
    app.radio[0].set_value("Rendimiento").run()
    assert not app.exception and len(app.metric) == 4
    app.radio[0].set_value("Errores").run()
    summary = json.loads(
        (ROOT / "experiments" / "error_summary_public.json").read_text(encoding="utf-8")
    )
    assert not app.exception and app.metric[0].value == str(summary["count"])
    app.radio[0].set_value("El modelo").run()
    assert not app.exception
    print("Streamlit público: hash, 4 predicciones, límite y 3 vistas verificados.")


if __name__ == "__main__":
    main()
