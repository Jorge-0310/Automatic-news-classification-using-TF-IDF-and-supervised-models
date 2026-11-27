# NewsScope

Clasificación temática experimental de noticias en inglés con TF-IDF y modelos supervisados.
El sistema predice una de cuatro clases de AG News: `World`, `Sports`, `Business` o `Sci/Tech`.
No verifica la veracidad de las noticias ni reemplaza la revisión humana.

## Estado

El repositorio contiene la línea base histórica, un script comparativo con
`LogisticRegression`, `ComplementNB` y `SGDClassifier`, y un prototipo local
Streamlit. En la ejecución registrada en `experiments/`, ganó regresión
logística con bigramas por F1 macro en validación (0,9192); en prueba obtuvo
F1 macro 0,9192 y accuracy 0,9193. El modelo guardado no se incluye en Git:
es necesario entrenarlo localmente. Véase [el plan de trabajo](docs/Plan%20de%20acci%C3%B3n%20de%203%20d%C3%ADas%20%E2%80%94%20NewsScope.md).

## Reproducción

Se recomienda Python 3.11, conexión a Internet para la primera descarga de AG News
y ejecutar desde la raíz del repositorio. En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py
```

En Windows use `.venv\Scripts\Activate.ps1` en vez de `source ...`; si no
puede activar el entorno, llame a `.venv\Scripts\python.exe` para instalar
dependencias y ejecutar el script. `requirements.txt` fija también PyArrow,
dependencia de `datasets`, para evitar combinaciones incompatibles. Se
comprobó instalación limpia y `pip check` en Windows/Python 3.11; no se ha
verificado en todas las plataformas.

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

Tras entrenar, pruebe el prototipo con `streamlit run app.py`. El modelo
local está en `models/newsscope.joblib` (excluido de Git): **primero** hay
que ejecutar `src/train.py` en cada clon. No cargue archivos joblib ajenos.
Para candidatos de errores reales: `python src/error_analysis.py`, que crea
`experiments/errors.csv` local, excluido de Git. Es indispensable clasificar
manualmente los casos; las columnas de causa vienen vacías. Existe un
borrador de IA para 12 de 20 ejemplos (`experiments/error_review_draft.json`)
que puede aplicarse con `python src/annotate_errors.py` y **debe validarse
por el equipo** con `python src/review_errors.py` (revisión interactiva de
al menos diez casos, solo local). Para las pruebas unitarias:
`python -m unittest discover -s tests -v`.
Para verificar la app con un modelo recién entrenado (cuatro entradas,
probabilidades y texto vacío): `python tests/smoke_app.py`.

## Estructura

- `src/baseline.py`: línea base ejecutable.
- `src/train.py`: selección en validación y prueba final del ganador.
- `src/error_analysis.py`: candidatos para análisis manual.
- `src/annotate_errors.py`: aplica borradores de anotación al CSV local.
- `src/config.py`: semilla compartida para los scripts de entrenamiento.
- `app.py`: clasificación local de una noticia.
- `data/README.md`: fuente y condiciones de almacenamiento de los datos.
- `notebooks/exploracion.ipynb`: exploración agregada sin publicar textos.
- `experiments/README.md`: archivos de resultados y procedimiento.
- `results/`: resultados preliminares ya versionados.
- `docs/`: propuesta, avance, bitácora y plan.

AG News está en inglés y no representa automáticamente el contexto periodístico
en español. Consulte [data/README.md](data/README.md) y
[docs/ia_responsable.md](docs/ia_responsable.md) antes de usar o compartir datos.

