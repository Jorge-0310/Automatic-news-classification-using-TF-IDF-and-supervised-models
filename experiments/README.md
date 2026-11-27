# Experimentos

El entrenamiento completo se ejecuta desde la raíz del repositorio con:

python src/train.py

El script descarga el conjunto de datos AG News, construye los cuatro
pipelines experimentales, mide su desempeño y guarda automáticamente los
resultados en la carpeta `experiments/`.

## Archivos generados

La ejecución produce los siguientes archivos:

experiments/
├── carbon/
│   ├── logistic_unigrams_validation.csv
│   ├── logistic_bigrams_validation.csv
│   ├── complement_nb_bigrams_validation.csv
│   ├── sgd_bigrams_validation.csv
│   ├── logistic_bigrams_repeat_validation.csv
│   └── final_model.csv
├── model_comparison.csv
├── stability.csv
├── run_config.json
├── confusion_matrix.png
└── f1_per_class.png

El archivo `logistic_bigrams_repeat_validation.csv` se genera con el modelo
ganador y la segunda semilla experimental. Su nombre se adapta
automáticamente si otra configuración resulta ganadora.

El modelo serializado queda fuera de Git en:

models/newsscope.joblib

Los resultados preliminares anteriores permanecen en `results/`, pero no se
utilizan para seleccionar el ganador del experimento principal.

## Diseño experimental

El script utiliza `RANDOM_STATE` desde `src/config.py` y aplica la misma
partición estratificada a los cuatro pipelines:

- 96.000 ejemplos para ajuste;
- 24.000 ejemplos para validación;
- 7.600 ejemplos para prueba final.

TF-IDF se ajusta dentro de cada `Pipeline`, después de separar los datos. De
esta manera, la información de validación y prueba no se utiliza para ajustar
el vocabulario durante el entrenamiento.

Las configuraciones comparadas son:

1. Regresión Logística con unigramas.
2. Regresión Logística con unigramas y bigramas.
3. ComplementNB con unigramas y bigramas.
4. SGDClassifier con pérdida `log_loss` y unigramas y bigramas.

El ganador se selecciona exclusivamente mediante F1 macro sobre el conjunto de
validación. La segunda ejecución utiliza otra semilla para comprobar la
estabilidad del ganador, pero no modifica la selección del modelo.

Después de seleccionar el ganador, el pipeline se reentrena con los 120.000
ejemplos del conjunto de entrenamiento y se evalúa una sola vez sobre los
7.600 ejemplos del conjunto de prueba.

## Resultados de rendimiento

`model_comparison.csv` contiene, para cada configuración:

- accuracy;
- F1 macro;
- precisión, recall y F1 por clase;
- tiempo de entrenamiento;
- tiempo de predicción;
- número de características generadas por TF-IDF;
- configuración de hiperparámetros;
- semilla utilizada.

`stability.csv` contiene el resultado del ganador con la semilla principal y
con la segunda semilla. La segunda fila también incluye, cuando la ejecución
finalizó correctamente, sus mediciones de energía y emisiones.

`run_config.json` conserva:

- la configuración del dataset;
- la semilla aleatoria;
- los tamaños de los conjuntos;
- las configuraciones comparadas;
- el ganador;
- la métrica utilizada para seleccionarlo;
- las métricas finales sobre prueba;
- la matriz de confusión;
- el costo computacional del modelo final.

## Costo computacional y huella de carbono

El proyecto utiliza `CodeCarbon` mediante
`OfflineEmissionsTracker`. La dependencia y su versión se encuentran en
`requirements.txt`.

Para cada configuración se registra una medición ambiental en:

experiments/carbon/

También se mide por separado:

- la segunda ejecución del ganador;
- el modelo final reentrenado con los 120.000 ejemplos.

Los archivos CSV de CodeCarbon incluyen, entre otros campos:

- duración de la ejecución;
- energía total consumida en kWh;
- emisiones estimadas en kg de CO2 equivalente;
- energía de CPU;
- energía de GPU;
- energía de memoria RAM;
- utilización de CPU, GPU y RAM;
- hardware utilizado;
- versión de Python;
- versión de CodeCarbon;
- sistema operativo;
- país configurado para la estimación.

La configuración ambiental utiliza:

country_iso_code="COL"
tracking_mode="process"
measure_power_secs=1

Las emisiones y el consumo energético son estimaciones. Pueden variar según el
hardware, el sistema operativo, la carga del equipo, la duración del proceso,
la versión de las dependencias y la intensidad de carbono utilizada por
CodeCarbon. Por esta razón, los valores deben interpretarse como aproximados y
no como mediciones físicas exactas.

En `model_comparison.csv`, las columnas ambientales principales son:

energy_kwh
emissions_kg_co2eq
cpu_energy_kwh
ram_energy_kwh
gpu_energy_kwh
feature_count
carbon_measurement_file

En `run_config.json`, los resultados del modelo final se almacenan dentro de:

"final_computational_cost"

Este bloque incluye:

train_seconds
predict_seconds
energy_kwh
emissions_kg_co2eq
cpu_energy_kwh
ram_energy_kwh
gpu_energy_kwh
carbon_file
feature_count
model_size_mb

