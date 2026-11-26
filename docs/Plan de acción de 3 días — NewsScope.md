# Plan de acción de 3 días — NewsScope

## Diagnóstico al iniciar

El proyecto está bien encaminado conceptualmente hacia la Ruta C de aprendizaje automático supervisado. El Hito 1 ya define el problema, la pregunta, la línea base de regresión logística, los modelos propuestos (`ComplementNB` y `SGDClassifier`), la comparación con unigramas/bigramas y la intención de usar validación cruzada estratificada. El Hito 2 implementa únicamente la línea base.

El repositorio actual contiene `src/baseline.py`, `results/`, `docs/avance_hito_2.md` y `docs/bitacora_ia.md`, pero todavía no contiene `README.md` completo, versiones fijadas, `data/`, `notebooks/`, `experiments/`, modelos comparativos, prototipo, matriz de confusión, análisis de diez errores ni informe IEEE. La línea base sí ejecuta de extremo a extremo: en la validación realizada produjo accuracy 0.9178 y F1 macro 0.9176 con 120.000 ejemplos de entrenamiento y 7.600 de prueba.

La instrucción final exige un informe IEEE de 6–8 páginas, repositorio reproducible, prototipo funcional y bitácora IA. Los criterios 1–8 suman 85 puntos; el criterio 9 corresponde a la sustentación individual y no se puede resolver solo con archivos.

## Orden de prioridad

1. **Criterio 5 — Experimentación y evaluación (15 puntos):** implementar la comparación mínima comprometida en el Hito 1.
2. **Criterio 4 — Implementación y reproducibilidad (15 puntos):** asegurar que otra persona pueda clonar, instalar y ejecutar el proyecto.
3. **Criterios 1–3 (30 puntos):** trasladar al informe la formulación, los datos y la justificación técnica.
4. **Criterio 6 — Análisis crítico (10 puntos):** generar y clasificar al menos diez errores reales.
5. **Criterio 7 — IA responsable (10 puntos):** documentar riesgos específicos y mitigaciones.
6. **Criterio 8 — Comunicación y bitácora (5 puntos):** cerrar el artículo IEEE y la bitácora con verificación y reflexión.

No conviene intentar mejorar el modelo con técnicas nuevas después de tener una comparación sólida. Es preferible una solución completa, reproducible y honesta que una búsqueda de hiperparámetros incompleta.

## Día 1 — Convertir la propuesta en experimentos reproducibles

### Objetivo de cierre
Tener un pipeline único que descargue AG News, entrene la línea base y compare al menos `LogisticRegression`, `ComplementNB` y `SGDClassifier` con las mismas particiones y métricas.

### Bloque 1: estructura y reproducibilidad

- Renombrar `README` a `README.md` y documentar la ejecución en menos de cinco comandos.
- Fijar versiones en `requirements.txt` (`datasets`, `scikit-learn`, `numpy`, `pandas`, `joblib`, `streamlit` o `gradio`).
- Crear `data/README.md` explicando que AG News se descarga mediante `datasets` y no se redistribuyen los textos.
- Crear `experiments/` para scripts, configuración, JSON/CSV de resultados, semillas y figuras.
- Definir `RANDOM_STATE=42` en un único archivo de configuración.
- Añadir `.gitignore` para cachés, modelos grandes y archivos descargados automáticamente.

### Bloque 2: implementación mínima de la comparación

Crear `src/train.py` o `src/experiment.py` con estas fases:

1. Cargar AG News.
2. Separar una validación estratificada del conjunto de entrenamiento, o usar `GridSearchCV` estratificado de 3 pliegues si el tiempo de cómputo lo permite.
3. Ajustar el vectorizador dentro de cada `Pipeline`, nunca antes de la partición.
4. Evaluar con `accuracy`, `f1_macro`, precision y recall por clase.
5. Guardar cada resultado con modelo, vectorizador, hiperparámetros, semilla, tiempo de entrenamiento y métricas.
6. Reservar el conjunto de prueba para la medición final; no usarlo para seleccionar hiperparámetros.

Experimentos mínimos recomendados:

- Regresión logística + TF-IDF unigramas.
- Regresión logística + TF-IDF unigramas/bigramas.
- ComplementNB + TF-IDF unigramas/bigramas.
- SGDClassifier + TF-IDF unigramas/bigramas.

Si el tiempo es limitado, utilizar una búsqueda pequeña y explícita: `ngram_range` en `[(1,1),(1,2)]`, `min_df` en `[2,5]`, `sublinear_tf` en `[True,False]`; para logística `C` en `[0.5,1,2]`; para ComplementNB `alpha` en `[0.1,1]`; para SGD `loss` en `['hinge','log_loss']`, `alpha` en `[1e-5,1e-4]`.

### Bloque 3: resultados y control de calidad

- Ejecutar la línea base y la comparación completa.
- Guardar `experiments/model_comparison.csv` y un JSON con la configuración.
- Generar matriz de confusión del mejor modelo y gráfico de F1 por clase.
- Repetir al menos la mejor configuración una segunda vez para comprobar estabilidad.
- Verificar que el mejor modelo no se elija mirando accidentalmente el conjunto de prueba.

### Entregable del día 1
Código de entrenamiento, resultados comparables, figuras preliminares y README ejecutable.

## Día 2 — Prototipo, errores e informe técnico

### Objetivo de cierre
Tener una demostración funcional y un borrador completo del informe IEEE.

### Bloque 1: análisis de errores

Crear `src/error_analysis.py` que guarde como CSV al menos diez ejemplos mal clasificados con: texto, clase real, clase predicha, probabilidades o margen, y tipo de error. Clasificar manualmente los casos en categorías como:

- vocabulario compartido entre `Business` y `Sci/Tech`;
- noticias de `World` con términos deportivos o económicos;
- titulares demasiado cortos o ambiguos;
- artículos con varias temáticas;
- entidades, nombres propios o acrónimos desconocidos;
- dependencia de contexto que no aparece en el texto.

Para cada tipo escribir una hipótesis de causa y una limitación. No presentar los diez casos como una lista sin interpretación.

### Bloque 2: prototipo funcional

Implementar `app.py` con Streamlit. Debe permitir pegar el texto de una noticia, ejecutar el pipeline guardado y mostrar categoría predicha, probabilidades por clase y una advertencia: “clasificación temática experimental; no verifica veracidad ni reemplaza revisión humana”. Incluir un ejemplo de entrada y un modo de carga del modelo entrenado.

Probarlo desde cero:

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

Si Streamlit consume demasiado tiempo, entregar un CLI sólido (`python src/predict.py --text "..."`) y documentarlo como prototipo, pero la app web es preferible para la demostración.

### Bloque 3: contenido del informe IEEE

Redactar en este orden: resumen y palabras clave; introducción con usuario real y pregunta; trabajo relacionado con mínimo cinco referencias, incluyendo tres artículos científicos y Russell & Norvig; formulación PEAS y tarea formal; datos, licencia y particiones; método y arquitectura; experimentos; tabla comparativa; matriz de confusión; errores; IA responsable; limitaciones; conclusiones; contribuciones; declaración de uso de IA; referencias IEEE.

La conclusión debe responder literalmente si `ComplementNB` o `SGDClassifier` mejoran a la línea base y bajo qué configuración. Si ningún modelo mejora de forma relevante, esa también es una conclusión válida.

### Entregable del día 2
Prototipo demostrable, CSV de errores, figuras finales y primer borrador completo del artículo.

## Día 3 — Cierre, auditoría y defensa

### Objetivo de cierre
Entregar un repositorio limpio y un informe que no prometa más de lo que los experimentos demuestran.

### Bloque 1: IA responsable

Añadir una sección específica para este caso:

- El corpus está en inglés y puede contener sesgos temporales, geográficos y editoriales.
- Las cuatro clases no equivalen a una clasificación universal de temas periodísticos.
- El sistema puede amplificar errores en `Business` y `Sci/Tech`, que fueron las clases más débiles de la línea base.
- No debe usarse para moderación, priorización de personas, decisiones financieras ni verificación de noticias.
- No introducir textos con datos personales en el prototipo sin autorización.
- Mitigaciones: revisión humana, advertencia de alcance, reporte por clase, análisis de errores, no redistribución del corpus, registro de versión y semilla, y reevaluación con datos locales antes de uso real.

### Bloque 2: bitácora IA

Reescribir `docs/bitacora_ia.md` con las columnas exigidas: fecha, integrante, herramienta, propósito, qué se aceptó/modificó/descartó y cómo se verificó. Añadir los usos realizados durante estos tres días y una reflexión de media página sobre al menos una sugerencia incorrecta de la IA, por ejemplo: detectar que una recomendación de evaluación podría causar fuga de información o que un resultado documentado no coincidía exactamente con una nueva ejecución.

No inventar prompts, resultados ni verificaciones. Cada integrante debe poder explicar todo código aceptado.

### Bloque 3: auditoría de entrega

- Clonar el repositorio en una carpeta nueva o en un entorno limpio.
- Ejecutar los comandos del README.
- Confirmar que el script no depende de rutas absolutas ni archivos locales no versionados.
- Confirmar que `requirements.txt` tiene versiones.
- Confirmar que existen `data/`, `src/`, `experiments/`, `docs/bitacora_ia.md` y `README.md`.
- Comprobar que cada tabla del informe tiene un archivo de resultados que la respalda.
- Revisar que las cifras del informe coincidan con los JSON/CSV finales.
- Corregir ortografía, nombres de integrantes, autores, fechas y enlaces.
- Hacer commits pequeños y descriptivos; no reemplazar todo con un único commit final.
- Preparar una copia PDF del informe y un enlace público al repositorio.

### Bloque 4: preparación de sustentación

Cada integrante debe practicar una explicación de dos minutos sobre problema, datos, pipeline, métricas, mejor modelo y limitación. Preparar respuestas para:

- por qué F1 macro además de accuracy;
- por qué TF-IDF debe ajustarse solo con entrenamiento;
- diferencia entre ComplementNB, logística y SGD;
- efecto esperado de agregar bigramas;
- por qué el conjunto de prueba se usa una sola vez;
- qué significa una matriz de confusión;
- qué cambio ocurriría al modificar `C`, `alpha` o `ngram_range`;
- cómo se detectó y corrigió un posible error de IA generativa.

### Entregable final del día 3
Informe IEEE final, repositorio completo, prototipo probado, bitácora IA completa, resultados consistentes y guion de sustentación.

## Mapa de cumplimiento de los criterios 1–8

| Criterio | Evidencia que debe existir al entregar | Riesgo actual |
|---|---|---|
| 1. Problema y formulación | Introducción, usuario, evidencia de relevancia, PEAS y tarea formal | Medio: está en Hito 1, falta integrarlo y respaldarlo mejor |
| 2. Datos | Fuente, licencia, tamaño, clases, partición, preprocesamiento y ausencia de fuga | Medio: falta licencia/documentación completa |
| 3. Fundamento técnico | Russell & Norvig, mínimo tres artículos científicos, alternativas y justificación | Alto: prácticamente pendiente |
| 4. Implementación | README.md, versiones fijadas, estructura exigida, ejecución limpia y commits | Alto: faltan archivos y estructura |
| 5. Evaluación | Línea base, dos modelos, unigramas/bigramas, búsqueda/validación, métricas y resultados | Crítico: solo existe la línea base |
| 6. Análisis crítico | Diez errores reales, categorías, hipótesis, matriz de confusión y límites | Crítico: pendiente |
| 7. IA responsable | Riesgos del corpus y del uso, impactos y mitigaciones concretas | Alto: solo hay limitaciones generales |
| 8. Comunicación y bitácora | Artículo IEEE, bitácora con verificación, reflexión y declaración IA | Alto: bitácora actual no tiene todas las columnas exigidas |

## Regla de recorte si el tiempo se agota

No sacrificar reproducibilidad, comparación mínima ni análisis de errores. Recortar primero la amplitud de la búsqueda de hiperparámetros, la cantidad de figuras y las mejoras no comprometidas. Es mejor reportar cuatro experimentos bien documentados que prometer una optimización que no pueda reproducirse.

## Fuentes de trabajo

[1]: https://github.com/Jorge-0310/Automatic-news-classification-using-TF-IDF-and-supervised-models "Repositorio público de NewsScope"
[2]: https://huggingface.co/datasets/fancyzhx/ag_news "AG News en Hugging Face"
[3]: https://arxiv.org/abs/1509.01626 "Character-level Convolutional Networks for Text Classification"
