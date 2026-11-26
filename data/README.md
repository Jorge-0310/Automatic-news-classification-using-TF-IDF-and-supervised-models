# Datos

La línea base obtiene [AG News (`fancyzhx/ag_news`)](https://huggingface.co/datasets/fancyzhx/ag_news)
con `datasets.load_dataset("fancyzhx/ag_news")`. La primera ejecución necesita
Internet; `datasets` almacena su propia caché local (normalmente fuera del repositorio).
No hay que descargar ni copiar manualmente archivos a `data/`.

El corpus contiene 120 000 noticias de entrenamiento y 7 600 de prueba,
etiquetadas como `World`, `Sports`, `Business` y `Sci/Tech`. Los textos están
en inglés. **No se redistribuyen los textos del corpus en este repositorio**:
los derechos de los artículos y las condiciones de uso deben comprobarse en
la [ficha del dataset](https://huggingface.co/datasets/fancyzhx/ag_news) y
en las fuentes originales antes de reutilizarlos. Los archivos descargados
localmente en `data/` están excluidos de Git.

La semilla de los scripts está en `src/config.py`; el conjunto de prueba no
debe utilizarse para escoger modelos ni hiperparámetros.
