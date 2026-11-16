# Hito 2. Avance del proyecto

## Datos preparados

Se integró el dataset **AG News** desde Hugging Face mediante la biblioteca `datasets`. El conjunto contiene 120.000 ejemplos de entrenamiento y 7.600 ejemplos de prueba, distribuidos en cuatro clases: `World`, `Sports`, `Business` y `Sci/Tech`.

Los datos contienen el texto de cada noticia y su respectiva etiqueta. El corpus está escrito en inglés, por lo que esta característica se reconoce como una limitación: los resultados no se pueden generalizar directamente a noticias en español ni al contexto periodístico colombiano.

## Línea base implementada

Se implementó una línea base de **Regresión Logística** utilizando una representación **TF-IDF con unigramas**. El vectorizador y el clasificador se integraron mediante un `Pipeline` de scikit-learn, con el fin de mantener el mismo flujo de procesamiento y evitar fuga de información.

La configuración inicial utilizada fue:

- `ngram_range=(1, 1)`;
- `min_df=2`;
- `max_df=0.95`;
- `sublinear_tf=True`;
- `max_iter=1000`;
- `random_state=42`.

La ejecución se realiza mediante:

```bash
python src/baseline.py
```

## Primer resultado

La línea base ya puede entrenarse y generar predicciones sobre el conjunto de prueba. Los valores de `accuracy` y `F1 macro` quedan registrados automáticamente en `results/baseline_results.json`, mientras que las métricas por clase quedan guardadas en `results/baseline_classification_report.txt`.

Estos resultados son preliminares y sirven para comprobar que el flujo de datos, entrenamiento, predicción y evaluación funciona correctamente. Todavía no representan la comparación final del proyecto.

## Reproducibilidad

El proyecto incluye las dependencias necesarias en `requirements.txt`. La ejecución se realiza con Python y scikit-learn, utilizando una semilla fija (`random_state=42`) en el modelo. El código fuente de la línea base se encuentra en `src/baseline.py`.

Para reproducir el experimento se deben instalar las dependencias y ejecutar:

```bash
python -m pip install -r requirements.txt
python src/baseline.py
```

## Próximos pasos

1. Implementar el modelo `ComplementNB`.
2. Implementar `SGDClassifier`.
3. Comparar los modelos bajo las mismas condiciones experimentales.
4. Comparar TF-IDF con unigramas frente a unigramas más bigramas.
5. Analizar la matriz de confusión y las métricas por clase.
6. Examinar al menos diez errores de clasificación.
7. Desarrollar el prototipo funcional del sistema.
