# NewsScope — borrador de contenido para artículo IEEE


## Resumen

Se estudia la clasificación temática de noticias en inglés de AG News en
cuatro categorías mediante TF-IDF y tres clasificadores supervisados:
regresión logística, Complement Naive Bayes y SGDClassifier. Una partición
estratificada del entrenamiento permite seleccionar por F1 macro sin usar
prueba. La comparación incluye unigramas frente a unigramas y bigramas;
se reportan tiempos y métricas por clase. En validación, la mejor
configuración fue regresión logística con unigramas y bigramas
(F1 macro 0,9192). Reentrenada, alcanzó F1 macro 0,9192 y accuracy
0,9193 en prueba; ComplementNB y SGDClassifier no mejoraron la
regresión logística en la validación de estas configuraciones.

**Palabras clave:** clasificación de texto, AG News, TF-IDF, regresión
logística, Complement Naive Bayes, SGD.

## I. Introducción y formulación

Un portal de noticias o sistema de monitoreo recibe textos cuyo tema necesita
organizar. Como indicador de escala del problema (no de demanda actual), el
corpus original de AG reúne más de un millón de artículos de más de dos mil
fuentes [7]; el subconjunto de clasificación contiene 127 600 ejemplos [6].
NewsScope propone una etiqueta inicial `World`, `Sports`,
`Business` o `Sci/Tech` para una noticia en inglés, sujeta a revisión humana;
no verifica hechos. La pregunta es si ComplementNB o SGDClassifier mejoran la
regresión logística de referencia y si añadir bigramas modifica el rendimiento
y los tipos de error.

**PEAS:** rendimiento: F1 macro (principal), accuracy, precision y recall por
clase, tiempo y límites de uso; entorno: noticias escritas en inglés con cuatro
etiquetas; actuadores: etiqueta y estimaciones de probabilidad en la
interfaz local; sensores: texto introducido por el usuario. Tarea formal:
dado un texto \(x\), predecir \(\hat y \in \{0,1,2,3\}\) a partir de un
pipeline entrenado exclusivamente con ejemplos etiquetados.

## II. Antecedentes y elección técnica

TF-IDF transforma cada documento en características ponderadas por frecuencia
local e inversa de documento; la normalización y los bigramas pueden ayudar a
capturar expresiones, a costa de mayor dimensionalidad. La regresión logística
es una referencia lineal interpretable; ComplementNB es una alternativa
probabilística para texto [2]; SGDClassifier ajusta un modelo lineal mediante
optimización estocástica. Los modelos lineales para clasificación de texto y
el marco de aprendizaje automático se describen en [3]–[5]. AG News ha sido
empleado para evaluar clasificadores de texto [1].

## III. Datos y método

AG News (`fancyzhx/ag_news`) ofrece 120 000 ejemplos de entrenamiento y
7 600 de prueba, con cuatro clases de 1 900 ejemplos cada una en prueba.
El corpus está en inglés. La ficha de Hugging Face señala licencia
**desconocida** [6]. El sitio del corpus original permite investigación
no comercial, solicita atribución y advierte que el copyright de los
artículos pertenece a las fuentes periodísticas [7]. La licencia de la
versión derivada no está confirmada; por precaución no se publican textos
ni errores con fragmentos y se debe revisar cualquier uso adicional.
Un 20 % estratificado de entrenamiento se reserva
para validación con semilla 42. Se construye un `Pipeline` separado por
configuración: TF-IDF (`min_df=2`, `max_df=0.95`, `sublinear_tf=True`)
y clasificador. Configuraciones: logística con unigramas, logística con
unigramas/bigramas, ComplementNB con unigramas/bigramas y SGDClassifier
con unigramas/bigramas. Los hiperparámetros exactos y tiempos se conservan
en `experiments/model_comparison.csv` y `run_config.json`.

Se selecciona únicamente con F1 macro de validación y se repite el ganador
con semilla 43 sobre la misma partición. Luego se ajusta de nuevo con
entrenamiento completo y se evalúa una sola vez en prueba. La línea base
histórica en `results/` ya había examinado prueba y **no** forma parte de la
selección ciega del experimento comparativo. No se realizó una búsqueda
exhaustiva ni validación cruzada de cinco pliegues; estas diferencias
respecto de la propuesta original reflejan una reducción de alcance:
cuatro configuraciones fijas y una sola partición para limitar el tiempo
de cómputo. No se evaluaron otros valores de `C`, `alpha` o `loss`,
por lo que no se puede concluir que una familia de modelos sea
universalmente superior.

## IV. Resultados y discusión

Resultados **de validación** (`experiments/model_comparison.csv`; 24 000
ejemplos, 6 000 por clase; tiempos de esta ejecución en segundos):

| Pipeline | F1 macro | Accuracy | Entrenamiento | Predicción |
|---|---:|---:|---:|---:|
| Logística + unigramas | 0,9172 | 0,9175 | 15,15 | 0,74 |
| Logística + unigramas/bigramas | **0,9192** | **0,9194** | 69,39 | 1,50 |
| ComplementNB + unigramas/bigramas | 0,9138 | 0,9142 | 7,86 | 1,61 |
| SGDClassifier (`log_loss`) + unigramas/bigramas | 0,8922 | 0,8927 | 9,43 | 1,50 |

Los bigramas mejoraron F1 macro de logística en validación aproximadamente
0,19 puntos porcentuales, con más tiempo de entrenamiento. La segunda
ejecución del ganador con semilla 43 dio el mismo F1 (0,9192): era
regresión logística y con los mismos datos/partición no se observaron
variaciones; **esto no demuestra estabilidad frente a otras particiones**.

Ganador seleccionado exclusivamente en validación: `logistic_bigrams`
(`C=1`, TF-IDF `ngram_range=(1,2)`). Tras reentrenarlo con 120 000 noticias,
su resultado **en prueba** (7 600) fue accuracy 0,9193, F1 macro 0,9192.
La tabla procede de `experiments/run_config.json`:

| Clase | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| World | 0,9399 | 0,9047 | 0,9220 | 1900 |
| Sports | 0,9516 | 0,9821 | 0,9666 | 1900 |
| Business | 0,8936 | 0,8842 | 0,8889 | 1900 |
| Sci/Tech | 0,8922 | 0,9063 | 0,8992 | 1900 |

La matriz de confusión (filas reales, columnas predichas en orden World,
Sports, Business, Sci/Tech) tiene diagonal 1719, 1866, 1680 y 1722.
Entre los pares más confundidos están 151 noticias de Business predichas
como Sci/Tech y 112 de Sci/Tech como Business. Véanse
`experiments/confusion_matrix.png` y `experiments/f1_per_class.png`.

La línea base histórica registró F1 macro 0,9177 en prueba; la diferencia
aproximada con el ganador es **0,15 puntos porcentuales**, inferior al
umbral de 1 punto del Hito 1. Esa referencia ya había usado prueba antes
del experimento, por lo que la comparación principal es la de validación.
No se calcularon intervalos bootstrap ni McNemar; no se afirma
significancia estadística.

### Costo computacional y huella de carbono

Además del desempeño predictivo, se registró el costo computacional de cada
configuración mediante `CodeCarbon` y `OfflineEmissionsTracker`. La medición
se realizó en modo `process`, con `country_iso_code="COL"` y un intervalo de
medición de un segundo. Los resultados se almacenaron en
`experiments/carbon/` y se resumieron en `model_comparison.csv` y
`run_config.json`.

La configuración seleccionada, Regresión Logística con TF-IDF y unigramas y
bigramas, consumió aproximadamente `0,0004575 kWh` y generó
`0,0000855 kg CO2eq` durante la ejecución de validación sobre 96.000 ejemplos
de entrenamiento y 24.000 ejemplos de validación. Esta configuración requirió
aproximadamente `68,20 segundos` de entrenamiento y `1,46 segundos` de
predicción sobre el conjunto de validación, y generó `383.016 características`
TF-IDF.

Al comparar las configuraciones, Regresión Logística con unigramas presentó un
menor costo ambiental, con un consumo aproximado de `0,0000869 kWh` y
emisiones de `0,0000162 kg CO2eq`. Sin embargo, su F1 macro fue `0,9172`,
inferior al `0,9193` obtenido por Regresión Logística con unigramas y
bigramas. ComplementNB con bigramas obtuvo un F1 macro de `0,9138`, un consumo
de `0,0000524 kWh` y emisiones de `0,0000098 kg CO2eq`. SGDClassifier con
bigramas obtuvo un F1 macro de `0,8922`, un consumo de `0,0000607 kWh` y
emisiones de `0,0000113 kg CO2eq`.

Después de seleccionar el ganador, el modelo se reentrenó con los 120.000
ejemplos del conjunto de entrenamiento. Esta ejecución final requirió
aproximadamente `93,44 segundos` de entrenamiento y `0,65 segundos` de
predicción sobre el conjunto de prueba. CodeCarbon estimó un consumo de
`0,0006542 kWh` y emisiones de `0,0001222 kg CO2eq`, equivalentes
aproximadamente a `0,1222 gramos de CO2eq`. El modelo final utilizó
`459.310 características` y ocupó aproximadamente `41,88 MB` en disco.

El extractor obtuvo 20 errores distintos, repartidos entre diferentes
pares de clases. Hay **12 borradores de categorización generados por IA**
en `experiments/error_review_draft.json`, con hipótesis y limitación por
caso, aplicados al CSV local; **todavía requieren confirmación humana**
tras leer cada texto. Se observan propuestas de categorías de noticias
tecnológicas con contenido financiero o empresarial (3 casos), noticias
internacionales con componente comercial o político y dos etiquetas World
para noticias olímpicas (posible ambigüedad de referencia). No se debe
afirmar que esas hipótesis expliquen causalmente las decisiones del modelo.
`experiments/errors.csv` incluye textos y no se distribuye.

## V. IA responsable y limitaciones

El corpus puede contener sesgos editoriales, temporales y geográficos.
Las clases no son una taxonomía universal. No extrapolar los resultados a
español ni a noticias actuales sin reevaluación. La aplicación no sirve
para moderación, decisiones financieras, verificación de veracidad o
priorización de personas. No ingresar datos personales sin autorización.
Ver medidas de mitigación concretas en `docs/ia_responsable.md`.

## VI. Conclusiones

Con las configuraciones evaluadas, **ni ComplementNB ni SGDClassifier
mejoraron a la regresión logística** en validación: 0,9138 y 0,8922 de
F1 macro frente a 0,9172 de la logística con unigramas y 0,9192 con
bigramas. Los bigramas aportaron una mejora pequeña (0,19 puntos
porcentuales en validación) a mayor costo de entrenamiento. El modelo
seleccionado obtuvo F1 macro 0,9192 en prueba. No se satisfizo el
criterio de una mejora de al menos 1 punto porcentual sobre la línea
base histórica, y no hay evidencia de significancia estadística.

## Contribuciones y uso de IA

**Pendiente:** integrantes deben confirmar su contribución individual.
Se utilizó IA como apoyo para generar y revisar estructura, código y
documentación; los resultados requieren verificación humana. Consultar
`docs/bitacora_ia.md`; no atribuir la interpretación de ejemplos a IA.

## Referencias iniciales (revisar metadatos antes de entregar)

[1] X. Zhang, J. Zhao y Y. LeCun, “Character-level Convolutional Networks
for Text Classification”, *NeurIPS*, 2015.
https://arxiv.org/abs/1509.01626

[2] J. D. M. Rennie, L. Shih, J. Teevan y D. R. Karger, “Tackling the
Poor Assumptions of Naive Bayes Text Classifiers”, *ICML*, 2003.
https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf

[3] T. Joachims, “Text Categorization with Support Vector Machines:
Learning with Many Relevant Features”, *ECML*, 1998.
https://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf

[4] F. Pedregosa *et al.*, “Scikit-learn: Machine Learning in Python”,
*JMLR*, vol. 12, pp. 2825–2830, 2011.
https://jmlr.org/papers/v12/pedregosa11a.html

[5] S. Russell y P. Norvig, *Artificial Intelligence: A Modern Approach*,
4.ª ed., Pearson, 2020.

[6] Hugging Face, “AG News — fancyzhx/ag_news”.
https://huggingface.co/datasets/fancyzhx/ag_news

[7] A. Gulli, “AG's corpus of news articles”, Universidad de Pisa.
https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
