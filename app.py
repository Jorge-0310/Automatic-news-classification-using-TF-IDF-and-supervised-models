"""Prototipo local NewsScope: streamlit run app.py."""

from pathlib import Path

import joblib
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "newsscope.joblib"
CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")
EXAMPLE = "A new spacecraft mission will study distant planets using an advanced telescope."

st.set_page_config(page_title="NewsScope", page_icon="📰")
st.title("NewsScope")
st.warning(
    "Clasificación temática experimental; no verifica veracidad ni reemplaza revisión humana. "
    "Evite introducir datos personales sin autorización."
)
st.caption("Solo noticias en inglés. Categorías: World, Sports, Business, Sci/Tech.")

if not MODEL_PATH.exists():
    st.error("No hay modelo local. Ejecute primero `python src/train.py`.")
    st.stop()


@st.cache_resource
def load_model():
    # Los archivos joblib pueden ejecutar código: usar únicamente artefactos
    # generados localmente por src/train.py, nunca archivos externos.
    return joblib.load(MODEL_PATH)


text = st.text_area("Texto de una noticia", value=EXAMPLE, height=180)
if st.button("Clasificar"):
    if not text.strip():
        st.error("Introduzca el texto de una noticia.")
    else:
        model = load_model()
        label = int(model.predict([text])[0])
        st.success(f"Categoría predicha: {CLASS_NAMES[label]}")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            st.subheader("Probabilidades estimadas por clase")
            for code, probability in sorted(
                zip(model.classes_, probabilities), key=lambda item: -item[1]
            ):
                st.write(f"{CLASS_NAMES[int(code)]}: {probability:.2%}")
            st.caption("Estas probabilidades no garantizan calibración ni veracidad.")
        else:
            st.info("El modelo seleccionado no proporciona probabilidades.")
