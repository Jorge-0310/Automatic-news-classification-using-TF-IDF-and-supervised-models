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


def section_lead(kicker, heading, accent, description):
    """Textos fijos, nunca provenientes de usuarios ni del corpus."""
    st.markdown(f'<div class="ns-kicker">— {kicker}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<h1 class="ns-hero">{heading}<br><em>{accent}</em></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="ns-intro">{description}</p>', unsafe_allow_html=True)


def panel_heading(label, badge):
    st.markdown(
        f'<div class="ns-panel-marker"><span>{label}</span>'
        f'<small>{badge}</small></div>',
        unsafe_allow_html=True,
    )


def select_example(text):
    # Callbacks de Streamlit se ejecutan antes de instanciar los widgets.
    st.session_state.news_input = text
    st.session_state.pop("prediction", None)


st.set_page_config(
    page_title="NewsScope | Laboratorio de noticias",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# CSS local y constante. Todas las entradas de usuario y las cifras se muestran
# con componentes nativos de Streamlit (no se insertan en HTML).
st.markdown(
    f"<style>{(BASE / 'editorial.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="ns-header"><div class="ns-logo">▥ &nbsp;News<span>Scope</span></div>'
    '<div class="ns-header-label">LABORATORIO DE CLASIFICACIÓN · AG NEWS</div>'
    '<div class="ns-header-status">● &nbsp;CLASIFICACIÓN EXPERIMENTAL</div></div>',
    unsafe_allow_html=True,
)
nav_selection = st.radio(
    "Secciones",
    ("01   Clasificar", "02   Rendimiento", "03   Errores", "04   El modelo"),
    horizontal=True,
    label_visibility="collapsed",
    key="section",
)
section = nav_selection.split("   ", 1)[1]

metrics = read_public_json(METRICS_FILE)
errors = read_public_json(ERRORS_FILE)

if section == "Clasificar":
    section_lead(
        "ESTACIÓN 01 / INFERENCIA",
        "Encuentra el tema.",
        "Conoce sus límites.",
        "Introduce una noticia en inglés y explora cómo un modelo lineal la clasifica "
        "entre cuatro categorías. Una predicción no es un juicio editorial ni una "
        "verificación de hechos.",
    )
    if "news_input" not in st.session_state:
        st.session_state.news_input = ""
    if "limiter_token" not in st.session_state:
        st.session_state.limiter_token = secrets.token_urlsafe(18)
    left, right = st.columns((1.05, .95), gap="medium")
    with left:
        panel_heading("A / ENTRADA", "EN INGLÉS")
        with st.form("classification_form"):
            text = st.text_area(
                "Texto de la noticia", key="news_input", height=180,
                max_chars=MAX_TEXT,
                placeholder="Escribe o pega aquí un titular o fragmento de una noticia en inglés…",
            )
            st.caption("Tu texto se procesa en el servidor de la app y no se guarda en un historial.")
            submitted = st.form_submit_button("Clasificar noticia ↗", type="primary")
        st.markdown('<div class="ns-examples">PRUEBA CON UN EJEMPLO</div>', unsafe_allow_html=True)
        example_columns = st.columns(4, gap="small")
        for column, (name, example) in zip(example_columns, EXAMPLES.items()):
            column.button(
                name, use_container_width=True, on_click=select_example, args=(example,)
            )
    if submitted:
        text = (text or "").strip()
        if not text:
            st.session_state.prediction = ("warning", "Escribe primero una noticia en inglés.")
        elif len(text) > MAX_TEXT:
            st.session_state.prediction = ("error", f"Máximo {MAX_TEXT} caracteres.")
        elif not shared_limiter().allow(st.session_state.limiter_token):
            st.session_state.prediction = ("warning", "Límite temporal alcanzado. Espera un minuto e inténtalo de nuevo.")
        else:
            try:
                with st.spinner("Clasificando con el modelo local…"):
                    model = load_trusted_model()
                    probabilities = model.predict_proba([text])[0]
                    scores = {
                        CLASS_NAMES[int(code)]: float(value)
                        for code, value in zip(model.classes_, probabilities)
                    }
                st.session_state.prediction = ("success", scores)
            except Exception:
                # No mostrar excepciones: pueden contener una URL/token del proveedor.
                st.session_state.prediction = (
                    "error",
                    "No se pudo cargar el modelo verificado. Comprueba Secrets, el acceso "
                    "al repositorio privado y deploy/model.sha256.",
                )
    with right:
        panel_heading("B / LECTURA", "4 CATEGORÍAS")
        state, value = st.session_state.get("prediction", ("empty", None))
        if state == "success":
            predicted = max(value, key=value.get)
            st.caption("CATEGORÍA PREDICHA")
            st.markdown(f"## {predicted}")
            st.divider()
            st.markdown("**Probabilidad estimada** · por clase")
            for name, score in sorted(value.items(), key=lambda item: -item[1]):
                label, percent = st.columns((3, 1))
                label.write(name)
                percent.write(f"{score:.1%}")
                st.progress(min(100, max(0, round(score * 100))))
            st.caption("Las probabilidades no están calibradas automáticamente; el mayor valor no implica certeza.")
        elif state == "warning":
            st.warning(value)
        elif state == "error":
            st.error(value)
        else:
            st.caption("Esperando una noticia para clasificar.")
            st.markdown(
                '<div class="ns-placeholder" aria-hidden="true">'
                '<i>W</i><i>S</i><i>B</i><i>T</i></div>',
                unsafe_allow_html=True,
            )
        if not MODEL_FILE.is_file():
            st.caption("El modelo se descarga de un repositorio privado en la primera clasificación.")
    st.markdown(
        '<div class="ns-notice"><strong>! &nbsp; Alcance experimental.</strong> '
        'NewsScope clasifica temas; no verifica veracidad, no reemplaza revisión '
        'humana ni debe usarse para decisiones sobre personas. No introduzcas '
        'datos personales sin autorización.</div>',
        unsafe_allow_html=True,
    )

elif section == "Rendimiento":
    section_lead(
        "ESTACIÓN 02 / EVIDENCIA",
        "Los números,",
        "con contexto.",
        "Métricas leídas de los artefactos del experimento. El navegador no entrena "
        "ni reevalúa modelos.",
    )
    if not metrics:
        st.error("Faltan métricas públicas. Ejecuta python src/dashboard_data.py antes del despliegue.")
    else:
        a, b, c, d = st.columns(4, gap="small")
        a.metric("01 / ACCURACY · PRUEBA", f"{metrics['test']['accuracy']:.4f}")
        b.metric("02 / F1 MACRO · PRUEBA", f"{metrics['test']['f1_macro']:.4f}")
        c.metric(
            "03 / MODELO · VALIDACIÓN",
            "Logística + bigramas"
            if metrics["selected_id"] == "logistic_bigrams"
            else metrics["selected_id"],
        )
        d.metric(
            "04 / AG NEWS · ENTRENAMIENTO / PRUEBA",
            f"{metrics['train_count']:,} / {metrics['test_count']:,}",
        )
        left, right = st.columns((1.25, .85), gap="medium")
        with left:
            panel_heading("COMPARACIÓN / VALIDACIÓN", f"{metrics['validation_count']:,} EJEMPLOS")
            st.subheader("Cuatro configuraciones")
            st.caption("Los valores de esta tabla son de validación, no de prueba.")
            table = pd.DataFrame(metrics["comparison"])[["id", "f1_macro", "accuracy", "train_seconds"]]
            st.dataframe(
                table.rename(columns={
                    "id": "Pipeline", "f1_macro": "F1 macro", "accuracy": "Accuracy",
                    "train_seconds": "Ajuste (s)",
                }),
                use_container_width=True, hide_index=True,
            )
        with right:
            panel_heading("DESGLOSE / PRUEBA", f"{metrics['test_count']:,} EJEMPLOS")
            st.subheader("F1 por categoría")
            st.caption("El promedio global no muestra dónde se concentran las dificultades.")
            for name in metrics["classes"]:
                score = metrics["test"]["per_class"][name]["f1-score"]
                label, percent = st.columns((3, 1))
                label.write(name)
                percent.write(f"{score:.1%}")
                st.progress(min(100, max(0, round(score * 100))))
        st.markdown('<div class="ns-matrix-marker"></div>', unsafe_allow_html=True)
        panel_heading("MATRIZ DE CONFUSIÓN / PRUEBA", "CLASIFICADOR FINAL")
        st.subheader("¿Dónde se confunde?")
        st.caption("Filas: clase real. Columnas: clase predicha. Fuera de la diagonal: errores.")
        st.dataframe(
            pd.DataFrame(
                metrics["confusion_matrix"], index=metrics["classes"], columns=metrics["classes"]
            ),
            use_container_width=True,
        )
        st.info(metrics["methodology"])

elif section == "Errores":
    section_lead(
        "ESTACIÓN 03 / ANÁLISIS CRÍTICO",
        "También importa",
        "dónde falla.",
        "Los recuentos proceden de casos revisados por el equipo. No publicamos "
        "noticias ni fragmentos del conjunto de datos.",
    )
    if not errors or errors["status"] != "confirmed":
        st.info("No hay un resumen de errores confirmado por el equipo.")
    else:
        left, right = st.columns(2, gap="medium")
        with left:
            panel_heading("CASOS REVISADOS", "SIN TEXTOS")
            st.metric("Casos de error revisados", errors["count"])
            st.caption("Esta muestra no representa todos los errores del conjunto de prueba.")
            st.dataframe(
                pd.DataFrame(errors["pairs"]).rename(columns={
                    "true_class": "Clase real", "predicted_class": "Predicha", "count": "Casos",
                }),
                use_container_width=True, hide_index=True,
            )
        with right:
            panel_heading("INTERPRETACIÓN RESPONSABLE", "LÍMITES")
            st.subheader("Un error no tiene una sola causa.")
            for limitation in errors["limitations"]:
                st.write(f"→ {limitation}")
            st.caption("Las hipótesis por caso no prueban causalidad y no se publican.")

else:
    section_lead(
        "ESTACIÓN 04 / MÉTODO",
        "Una herramienta,",
        "no un verificador.",
        "NewsScope apoya la organización inicial de noticias en inglés. No decide "
        "su relevancia, intención ni veracidad.",
    )
    steps = st.columns(4, gap="small")
    for column, kicker, title, detail in zip(
        steps,
        ("01 / ENTRADA", "02 / REPRESENTACIÓN", "03 / MODELO", "04 / SALIDA"),
        ("Texto en inglés", "TF-IDF", "Clasificador lineal", "Clase + probabilidades"),
        ("Titular o fragmento breve.", "Unigramas y bigramas.",
         "Seleccionado por validación.", "Estimaciones, no certezas."),
    ):
        with column:
            panel_heading(kicker, "→")
            st.markdown(f"**{title}**")
            st.caption(detail)
    first, second = st.columns(2, gap="medium")
    with first:
        panel_heading("PREGUNTA", "CONTEXTO")
        st.subheader("¿Qué problema resuelve?")
        st.write(
            "Un sistema de monitoreo necesita asignar una primera categoría temática "
            "a noticias recibidas. Se evalúa por F1 macro y métricas de cada clase; "
            "la etiqueta siempre puede revisarse por humanos."
        )
    with second:
        panel_heading("ALTERNATIVAS", "MODELOS")
        st.subheader("Tres formas de aprender")
        st.write(
            "**Regresión logística:** modelo lineal discriminativo. "
            "**ComplementNB:** clasificador probabilístico para texto. "
            "**SGDClassifier:** modelo lineal ajustado por gradiente estocástico."
        )
    third, fourth = st.columns(2, gap="medium")
    with third:
        panel_heading("LÍMITE 01", "COBERTURA")
        st.subheader("Cuatro categorías")
        st.write(
            "World, Sports, Business y Sci/Tech no cubren todos los temas; "
            "una noticia puede tratar varios asuntos a la vez."
        )
    with fourth:
        panel_heading("LÍMITE 02", "GENERALIZACIÓN")
        st.subheader("Corpus histórico, en inglés")
        st.write(
            "Un resultado en AG News no se traslada directamente a noticias actuales, "
            "locales o en español. Las probabilidades no garantizan calibración."
        )
    st.markdown(
        '<div class="ns-kicker ns-cost-kicker">— MEDICIÓN / EXPERIMENTO REGISTRADO</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Costo computacional y huella de carbono")
    st.write(
        "El reentrenamiento final del modelo ganador utilizó los 120.000 ejemplos "
        "de AG News y se evaluó una vez sobre los 7.600 de prueba."
    )
    if metrics and metrics.get("final_computational_cost"):
        cost = metrics["final_computational_cost"]
        a, b, c, d = st.columns(4, gap="small")
        a.metric("AJUSTE FINAL", f"{cost['train_seconds']:.1f} s")
        b.metric(
            "ENERGÍA ESTIMADA",
            f"{cost['energy_kwh'] * 1000:.2f} Wh"
            if cost["energy_kwh"] is not None else "No disponible",
        )
        c.metric(
            "EMISIONES ESTIMADAS",
            f"{cost['emissions_kg_co2eq'] * 1000:.3f} g CO₂e"
            if cost["emissions_kg_co2eq"] is not None else "No disponible",
        )
        d.metric("TAMAÑO DEL MODELO", f"{cost['model_size_mb']:.1f} MiB")
        st.caption(
            f"La predicción del lote completo de {metrics['test_count']:,} noticias "
            f"de prueba tardó {cost['predict_seconds']:.2f} s; esto NO mide "
            "la latencia ni la huella de una petición en la web."
        )
        st.markdown('<div class="ns-cost-table"></div>', unsafe_allow_html=True)
        st.markdown("**Comparación de configuraciones · validación**")
        st.dataframe(
            pd.DataFrame([
                {
                    "Pipeline": row["id"],
                    "Ajuste (s)": round(row["train_seconds"], 1),
                    "Energía estimada (Wh)": (
                        round(row["energy_kwh"] * 1000, 3)
                        if row["energy_kwh"] is not None else None
                    ),
                    "Emisiones estimadas (g CO₂e)": (
                        round(row["emissions_kg_co2eq"] * 1000, 4)
                        if row["emissions_kg_co2eq"] is not None else None
                    ),
                }
                for row in metrics["comparison"]
            ]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "«Ajuste» mide solo el entrenamiento de cada configuración; "
            "CodeCarbon registra el proceso durante entrenamiento y evaluación. "
            "Son ejecuciones distintas de la final: no sumar ni interpretar "
            "estas filas como emisiones por noticia."
        )
        st.info(
            "CodeCarbon (modo offline, proceso, país configurado: Colombia) "
            "estima energía y CO₂ equivalente según hardware e intensidad "
            "eléctrica. Estas cifras no incluyen descarga de datos, "
            "preparación, almacenamiento ni funcionamiento continuo de "
            "Streamlit Cloud; no son mediciones físicas exactas ni una "
            "huella total del proyecto."
        )
    else:
        st.info("Las mediciones ambientales de esta ejecución no están disponibles.")
    st.info("Los textos de AG News no se redistribuyen en esta interfaz.")
    st.markdown(
        "[Repositorio y documentación académica]"
        "(https://github.com/Jorge-0310/Automatic-news-classification-using-TF-IDF-and-supervised-models)"
    )

st.markdown(
    '<div class="ns-footer"><strong>NewsScope</strong> · Un experimento de '
    'clasificación temática &nbsp; / &nbsp; EXPLORA LA EVIDENCIA. REVISA LOS LÍMITES.</div>',
    unsafe_allow_html=True,
)
