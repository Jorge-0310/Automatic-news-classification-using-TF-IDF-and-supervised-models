"""Prueba integrada del prototipo; ejecutar tras entrenar el modelo.

python tests/smoke_app.py
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "newsscope.joblib"


def main():
    if not MODEL.exists():
        raise SystemExit("Falta el modelo: ejecute primero python src/train.py")
    # En un entorno limpio, deserializar el pipeline puede superar los 3 s
    # predeterminados de AppTest aunque la aplicación funcione correctamente.
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception
    examples = (
        "The football team won the championship match after scoring three goals.",
        "Shares of the bank rose after quarterly earnings beat estimates.",
        "Researchers launched a spacecraft to study distant planets.",
        "A report about events in a city.",
    )
    for text in examples:
        app.text_area[0].set_value(text)
        app.button[0].click().run()
        assert not app.exception
        assert len(app.success) == 1
        assert len(app.subheader) == 1  # Probabilidades de las cuatro clases.
    app.text_area[0].set_value("")
    app.button[0].click().run()
    assert not app.exception
    assert len(app.error) == 1
    print("App: 4 entradas con probabilidades + texto vacío verificados.")


if __name__ == "__main__":
    main()
