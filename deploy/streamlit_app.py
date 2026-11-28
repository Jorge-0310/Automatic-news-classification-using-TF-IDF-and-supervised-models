"""Demo pública desde este repositorio, sin entrenamiento ni corpus."""

import hashlib
import json
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
METRICS_FILE = ROOT / "experiments" / "dashboard_metrics.json"
ERRORS_FILE = ROOT / "experiments" / "error_summary_public.json"
MODEL_FILE = ROOT / "models" / "newsscope.joblib"
CHECKSUM_FILE = BASE / "model.sha256"
MAX_TEXT = 5000
WINDOW_SECONDS = 60
PER_SESSION_LIMIT = 5
GLOBAL_LIMIT = 30
CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")
EXAMPLES = {
    "World": "World leaders signed a new international treaty after diplomatic talks.",
    "Sports": "The football team won the championship match after scoring three goals.",
    "Business": "The company reported higher quarterly profits and rising stock prices.",
    "Sci/Tech": "Scientists launched a spacecraft to explore distant planets with a telescope.",
}


class RequestLimiter:
    """Límite de cortesía en memoria; no sustituye protección del proveedor."""

    def __init__(self):
        self.lock = threading.Lock()
        self.global_calls = deque()
        self.sessions = defaultdict(deque)

    def allow(self, session_key):
        now = time.monotonic()
        with self.lock:
            while self.global_calls and now - self.global_calls[0] >= WINDOW_SECONDS:
                self.global_calls.popleft()
            for key, calls in list(self.sessions.items()):
                while calls and now - calls[0] >= WINDOW_SECONDS:
                    calls.popleft()
                if not calls:
                    del self.sessions[key]
            calls = self.sessions[session_key]
            if len(self.global_calls) >= GLOBAL_LIMIT or len(calls) >= PER_SESSION_LIMIT:
                return False
            self.global_calls.append(now)
            calls.append(now)
            return True


@st.cache_resource
def shared_limiter():
    return RequestLimiter()


@st.cache_resource
def load_trusted_model():
    """Cargar solo el artefacto cuyo hash se fijó en el repositorio público."""
    if not CHECKSUM_FILE.is_file():
        raise FileNotFoundError("Suma de verificación no disponible.")
    model_path = MODEL_FILE
    if not model_path.is_file():
        try:
            repo = st.secrets["HF_MODEL_REPO"]
            token = st.secrets["HF_TOKEN"]
        except (KeyError, FileNotFoundError) as exc:
            raise FileNotFoundError("Configure HF_MODEL_REPO y HF_TOKEN en Secrets.") from exc
        if not isinstance(repo, str) or not isinstance(token, str) or not repo or not token:
            raise ValueError("Configuración de modelo privado inválida.")
        # Nunca entrenar ni publicar el joblib en este repositorio público.
        model_path = Path(hf_hub_download(
            repo_id=repo, filename="newsscope.joblib", repo_type="model", token=token
        ))
    if model_path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("El archivo del modelo supera el tamaño esperado.")
    expected = CHECKSUM_FILE.read_text(encoding="ascii").strip().split()[0].lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("Suma SHA-256 inválida.")
    digest = hashlib.sha256()
    with model_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("El modelo no coincide con su suma SHA-256.")
    # SHA-256 no hace seguro un archivo ajeno: controlar quién puede escribir
    # en el repo privado y revisar personalmente cada actualización del hash.
    return joblib.load(model_path)


def read_public_json(path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as source:
        return json.load(source)


st.set_page_config(page_title="NewsScope | Clasificación de noticias", page_icon="📰", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background:#F8F7F4;color:#182433}
    .block-container {max-width:1150px;padding-top:2rem;padding-bottom:3rem}
    h1,h2,h3 {color:#182433;letter-spacing:-.025em}
    h1 {font-size:3rem!important}
    [data-testid="stSidebar"] {background:#EEF1F4;border-right:1px solid #D9DEE5}
    [data-testid="stMetricValue"] {color:#1E6F78}
    div.stButton > button[kind="primary"],div.stFormSubmitButton > button[kind="primary"] {
      background:#D95D39;border-color:#D95D39;color:white
    }
    div.stButton > button:hover,div.stFormSubmitButton > button:hover {border-color:#1E6F78}
    .editorial-kicker {color:#D95D39;font-size:.75rem;letter-spacing:.18em;font-weight:700}
    .editorial-note {color:#5E6974;font-size:.92rem}
    </style>
    """,
    unsafe_allow_html=True,  # Solo CSS constante; nunca incorporar texto de usuario aquí.
)

metrics = read_public_json(METRICS_FILE)
errors = read_public_json(ERRORS_FILE)
with st.sidebar:
    st.title("NewsScope")
    st.caption("LABORATORIO · AG NEWS")
    section = st.radio(
        "Explorar",
        ("Clasificar", "Rendimiento", "Errores", "El modelo"),
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Clasificación temática experimental. No verifica veracidad ni reemplaza revisión humana.")
    st.caption("No introduzcas datos personales sin autorización.")

if section == "Clasificar":
    st.markdown('<p class="editorial-kicker">01 / INFERENCIA</p>', unsafe_allow_html=True)
    st.title("Encuentra el tema. Conoce sus límites.")
    st.write("Pega una noticia en inglés para estimar su categoría entre World, Sports, Business y Sci/Tech.")
    st.warning(
        "Clasificación experimental: las probabilidades no están calibradas. "
        "No verifica veracidad, no reemplaza revisión humana y no debe utilizarse "
        "para moderación ni decisiones sobre personas."
    )
    st.caption(
        "La entrada llega al servidor de Streamlit que hospeda la app; este código "
        "no la registra ni la envía a otros servicios. Evita datos personales."
    )
    if "news_input" not in st.session_state:
        st.session_state.news_input = ""
    if "limiter_token" not in st.session_state:
        st.session_state.limiter_token = secrets.token_urlsafe(18)
    example_columns = st.columns(4)
    for column, (name, example) in zip(example_columns, EXAMPLES.items()):
        if column.button(name, use_container_width=True):
            st.session_state.news_input = example
            st.rerun()
    with st.form("classification_form"):
        text = st.text_area(
            "Texto de la noticia", key="news_input", height=180,
            max_chars=MAX_TEXT, placeholder="Write or paste an English news headline…",
        )
        submitted = st.form_submit_button("Clasificar noticia", type="primary")
    if submitted:
        text = (text or "").strip()
        if not text:
            st.warning("Escribe primero una noticia en inglés.")
        elif len(text) > MAX_TEXT:
            st.error(f"Máximo {MAX_TEXT} caracteres.")
        elif not shared_limiter().allow(st.session_state.limiter_token):
            st.warning("Límite temporal alcanzado. Espera un minuto e inténtalo de nuevo.")
        else:
            try:
                with st.spinner("Clasificando…"):
                    model = load_trusted_model()
                    probabilities = model.predict_proba([text])[0]
                    scores = {
                        CLASS_NAMES[int(code)]: float(value)
                        for code, value in zip(model.classes_, probabilities)
                    }
                predicted = max(scores, key=scores.get)
                st.success(f"Categoría predicha: {predicted}")
                st.subheader("Probabilidades estimadas")
                for name, value in sorted(scores.items(), key=lambda item: -item[1]):
                    label, track, percent = st.columns([1, 4, 1])
                    label.write(name)
                    track.progress(min(100, max(0, round(value * 100))))
                    percent.write(f"{value:.1%}")
                st.caption("Una probabilidad alta no implica que la noticia sea verdadera.")
            except Exception:
                # No mostrar la excepción: puede contener una URL/token del proveedor.
                st.error(
                    "No se pudo cargar el modelo verificado. Comprueba los Secrets, "
                    "el acceso al repositorio privado y deploy/model.sha256."
                )
    if not MODEL_FILE.is_file():
        st.caption("El modelo se descarga de un repositorio privado al clasificar por primera vez.")

elif section == "Rendimiento":
    st.markdown('<p class="editorial-kicker">02 / EVIDENCIA</p>', unsafe_allow_html=True)
    st.title("Los números, con contexto.")
    if not metrics:
        st.error("Faltan métricas públicas. Ejecuta python src/dashboard_data.py antes de preparar el despliegue.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Accuracy · prueba", f"{metrics['test']['accuracy']:.4f}")
        b.metric("F1 macro · prueba", f"{metrics['test']['f1_macro']:.4f}")
        c.metric(
            "Modelo elegido en validación",
            "Regresión logística + bigramas"
            if metrics["selected_id"] == "logistic_bigrams"
            else metrics["selected_id"],
        )
        d.metric("AG News · entrenamiento / prueba", f"{metrics['train_count']:,} / {metrics['test_count']:,}")
        st.caption(metrics["methodology"])
        st.subheader("Comparación en validación")
        table = pd.DataFrame(metrics["comparison"])[["id", "accuracy", "f1_macro", "train_seconds"]]
        st.dataframe(
            table.rename(columns={
                "id": "Configuración", "accuracy": "Accuracy", "f1_macro": "F1 macro",
                "train_seconds": "Tiempo de ajuste (s)",
            }),
            use_container_width=True, hide_index=True,
        )
        left, right = st.columns(2)
        with left:
            st.subheader("F1 por clase · prueba")
            per_class = metrics["test"]["per_class"]
            values = pd.DataFrame({
                "Clase": metrics["classes"],
                "F1": [per_class[name]["f1-score"] for name in metrics["classes"]],
            }).set_index("Clase")
            st.bar_chart(values)
        with right:
            st.subheader("Matriz de confusión · prueba")
            st.caption("Filas = clase real · columnas = clase predicha.")
            st.dataframe(
                pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=metrics["classes"],
                    columns=metrics["classes"],
                ),
                use_container_width=True,
            )
        st.caption("Las cifras están guardadas en experiments/. La interfaz no vuelve a entrenar modelos.")

elif section == "Errores":
    st.markdown('<p class="editorial-kicker">03 / ANÁLISIS CRÍTICO</p>', unsafe_allow_html=True)
    st.title("También importa dónde falla.")
    if not errors or errors["status"] != "confirmed":
        st.info("No hay un resumen de errores confirmado por el equipo.")
    else:
        st.metric("Casos revisados", errors["count"])
        st.caption(
            "Recuentos de una muestra declarada por el equipo, no de todos los "
            "errores de prueba. Nunca se cargan textos del corpus en esta app."
        )
        st.subheader("Pares de clases confundidos")
        st.dataframe(
            pd.DataFrame(errors["pairs"]).rename(columns={
                "true_class": "Clase real", "predicted_class": "Predicha", "count": "Casos",
            }),
            use_container_width=True, hide_index=True,
        )
        st.subheader("Limitaciones")
        for limitation in errors["limitations"]:
            st.write(f"• {limitation}")
        st.caption("Las hipótesis por caso no prueban causalidad y no se publican.")

else:
    st.markdown('<p class="editorial-kicker">04 / MÉTODO</p>', unsafe_allow_html=True)
    st.title("Una herramienta, no un verificador.")
    st.write("Texto en inglés → TF-IDF → clasificador lineal → categoría y probabilidades.")
    st.write(
        "Regresión logística aprende pesos discriminativos; ComplementNB es una "
        "alternativa probabilística para texto; SGDClassifier ajusta un modelo "
        "lineal mediante gradiente estocástico."
    )
    st.info(
        "AG News es histórico y está en inglés. Sus cuatro clases no cubren todos "
        "los temas periodísticos; el rendimiento no se traslada automáticamente "
        "a noticias actuales, locales o en español."
    )
    st.markdown(
        "[Repositorio y documentación académica]"
        "(https://github.com/Jorge-0310/Automatic-news-classification-using-TF-IDF-and-supervised-models)"
    )
