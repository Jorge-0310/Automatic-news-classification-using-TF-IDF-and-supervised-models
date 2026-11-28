# NewsScope

Clasificación temática experimental de noticias en inglés con TF-IDF y modelos supervisados.
El sistema predice una de cuatro clases de AG News: `World`, `Sports`, `Business` o `Sci/Tech`.
No verifica la veracidad de las noticias ni reemplaza la revisión humana.

## Estado

El repositorio contiene una web de demostración (FastAPI + HTML/CSS/JS), la
línea base histórica, la comparación con `LogisticRegression`, `ComplementNB`
y `SGDClassifier`, y Streamlit como alternativa local. En la ejecución
registrada en `experiments/`, ganó regresión logística con bigramas; en prueba
obtuvo F1 macro 0,9192 y accuracy 0,9193. 
## Reproducción

Se recomienda Python 3.11, conexión a Internet para la primera descarga de
AG News y ejecutar desde la raíz del repositorio. En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py
uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Abrir **http://127.0.0.1:8000/**. En Windows, sustituir el segundo comando
por `.venv\Scripts\Activate.ps1`; si PowerShell no permite la activación,
usar `.venv\Scripts\python.exe -m pip install -r requirements.txt`,
`.venv\Scripts\python.exe src/train.py` y
`.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log`.
Escuchar solo en `127.0.0.1`: la demostración no lleva autenticación y no
debe publicarse como servicio abierto. `requirements.txt` también fija PyArrow.

La línea base previa se puede repetir por separado con `python src/baseline.py`.
Escribe
`results/baseline_results.json` y `results/baseline_classification_report.txt`.
Sus resultados son **preliminares**, obtenidos con prueba antes de definir
la comparación: no participan en la selección. `src/train.py` descarga
`fancyzhx/ag_news`, separa 20 % del entrenamiento para validación estratificada,
ajusta TF-IDF dentro de cada pipeline, compara cuatro configuraciones por
F1 macro, repite la ganadora con otra semilla, reentrena sobre entrenamiento
completo y finalmente evalúa **solo la ganadora** sobre prueba.
No es una búsqueda extensa de hiperparámetros ni validación cruzada.

La interfaz principal se sirve desde `web/` por `api.py`. La API lee las
métricas de `experiments/run_config.json`, `model_comparison.csv` y
`stability.csv`; el frontend **no entrena** ni introduce cifras manuales.
`python src/dashboard_data.py` actualiza la copia pública opcional
`experiments/dashboard_metrics.json`. Esta copia incluye también el costo
computacional del modelo final y las estimaciones de CodeCarbon por
configuración, que la vista «El modelo» presenta con sus límites. No mide
el consumo del despliegue en Streamlit ni el de una predicción individual.
La vista de errores solo ofrece
agregados de `experiments/error_review_confirmed.json` cuando existe; no
expone textos, hipótesis redactadas por el revisor ni `errors.csv`.
La inferencia carga `models/newsscope.joblib` una vez y solo desde disco
local: nunca cargue archivos joblib ajenos. Si falta el artefacto, el
dashboard sigue mostrando métricas, pero clasificar devuelve 503.

Streamlit sigue disponible como respaldo: `streamlit run app.py`.
No se guarda el texto introducido en un historial ni se envía a servicios
externos; no introduzca datos personales sin autorización. El servidor se
inicia sin logs de acceso para reducir la exposición de entradas.

## Demo pública en Streamlit Community Cloud

La app completa `deploy/streamlit_app.py` se despliega **desde este mismo
repositorio público** seleccionándola como entrypoint en Community Cloud.
La interfaz de Streamlit es independiente del HTML de `web/`. Antes de
desplegar, compruebe que la rama seleccionada contiene todos los archivos
de configuración y estilos requeridos por `deploy/streamlit_app.py`.
No se entrena en Cloud ni se sube `models/newsscope.joblib` a GitHub: se
descarga de un repositorio *privado de modelos* con un token de solo lectura
guardado en Streamlit Secrets y se comprueba contra `deploy/model.sha256`.
El paso a paso (crear repositorio privado de modelos, configurar Secrets y
seleccionar la rama) está en **[deploy/README.md](deploy/README.md)**.

## Análisis de errores y pruebas
Para extraer candidatos de errores reales, se ejecuta:

```bash
python src/error_analysis.py
```

El comando crea `experiments/errors.csv` local, excluido de Git porque
contiene fragmentos de artículos del corpus. El archivo no se distribuye ni
se utiliza para publicar textos de noticias.

El repositorio incluye un borrador generado por IA para 12 de los 20 casos
(`experiments/error_review_draft.json`). Puede aplicarse de forma opcional
con `python src/annotate_errors.py`, pero esas anotaciones son provisionales:
deben validarse leyendo los textos mediante la revisión interactiva:

```bash
python src/review_errors.py
```

El revisor debe identificarse y confirmar o editar cada caso. El script no
guarda la confirmación si se revisan menos de 10 casos, no sobrescribe una
revisión confirmada existente y exporta únicamente índices, clases y
anotaciones, sin incluir los textos del corpus. Las hipótesis de causa deben
redactarse como hipótesis, no como hechos probados. En la revisión registrada
en `experiments/error_review_confirmed.json`, se confirmaron 10 casos; se
agruparon cualitativamente en cruces temáticos (4), asociaciones léxicas
insuficientes (3) y contexto o etiqueta insuficiente (3). 
Para ejecutar las pruebas unitarias:

```bash
python -m unittest discover -s tests -v
```

Para verificar la aplicación con un modelo recién entrenado, incluyendo
cuatro entradas, probabilidades y texto vacío:

```bash
python tests/smoke_app.py
```

Para verificar la web y la API con el modelo local:

```bash
python tests/smoke_frontend.py
```

También existe una comprobación de la interfaz pública de Streamlit:

```bash
python tests/smoke_streamlit_public.py
```

## Estructura

- `src/baseline.py`: línea base ejecutable.
- `src/train.py`: selección en validación y prueba final del ganador.
- `src/error_analysis.py`: candidatos para análisis manual.
- `src/annotate_errors.py`: aplica borradores de anotación al CSV local.
- `src/config.py`: semilla compartida para los scripts de entrenamiento.
- `api.py`: API FastAPI y servidor de la interfaz principal.
- `src/inference.py`: inferencia con carga única del modelo local.
- `src/dashboard_data.py`: métricas públicas y recuentos sin noticias.
- `web/`: frontend responsive de cuatro vistas, sin librerías JS externas.
- `app.py`: alternativa Streamlit.
- `deploy/streamlit_app.py`: interfaz pública de Streamlit y descarga segura del modelo privado.
- `data/README.md`: fuente y condiciones de almacenamiento de los datos.
- `notebooks/exploracion.ipynb`: exploración agregada sin publicar textos.
- `experiments/README.md`: archivos de resultados y procedimiento.
- `results/`: resultados preliminares ya versionados.
- `docs/`: propuesta, avance, bitácora y plan.

AG News está en inglés y no representa automáticamente el contexto periodístico
en español. Consulte [data/README.md](data/README.md) y
[docs/ia_responsable.md](docs/ia_responsable.md) antes de usar o compartir datos.
