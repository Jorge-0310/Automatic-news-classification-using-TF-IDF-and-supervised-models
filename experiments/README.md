# Experimentos

`python src/train.py` genera `model_comparison.csv` (validación),
`stability.csv` (segunda semilla del ganador), `run_config.json` (configuración,
selección y evaluación final), `confusion_matrix.png` y `f1_per_class.png`.
El modelo serializado queda fuera de Git en `models/newsscope.joblib`.
Los resultados preliminares anteriores siguen en `results/` y **no** fueron
utilizados para seleccionar el ganador.

El script usa `RANDOM_STATE` de `src/config.py` y la misma partición
estratificada 80/20 de los 120 000 ejemplos de entrenamiento para los cuatro
pipelines. TF-IDF se ajusta dentro de cada `Pipeline`, nunca antes de separar.
El ganador se decide por F1 macro en validación; la segunda semilla solo
comprueba estabilidad y **no cambia la selección**. Después se reentrena con
los 120 000 ejemplos y se evalúa una vez en prueba (7 600 ejemplos).

`python src/error_analysis.py` extrae candidatos mal clasificados del conjunto
de prueba a `experiments/errors.csv` **local**, excluido de Git por contener
fragmentos de artículos. El tipo, hipótesis causal y limitación se dejan vacíos
para revisión humana: no se deben presentar como análisis ya realizado.
Si el archivo local ya contiene anotaciones, el extractor se niega a
sobrescribirlo.
El borrador `error_review_draft.json` propone clasificaciones para 12 índices
sin copiar textos. `python src/annotate_errors.py` puede aplicarlo al CSV
local, conservando cualquier anotación distinta. **Estas clasificaciones de
IA son provisionales y deben ser validadas por el equipo** leyendo los textos.
