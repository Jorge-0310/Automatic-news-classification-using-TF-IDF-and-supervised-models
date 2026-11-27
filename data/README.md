# Datos

La línea base obtiene [AG News (`fancyzhx/ag_news`)](https://huggingface.co/datasets/fancyzhx/ag_news)
con `datasets.load_dataset("fancyzhx/ag_news")`. La primera ejecución necesita
Internet; `datasets` almacena su propia caché local (normalmente fuera del repositorio).
No hay que descargar ni copiar manualmente archivos a `data/`.

El corpus contiene 120 000 noticias de entrenamiento y 7 600 de prueba,
etiquetadas como `World`, `Sports`, `Business` y `Sci/Tech`. Los textos están
en inglés. La [ficha de Hugging Face](https://huggingface.co/datasets/fancyzhx/ag_news/raw/main/README.md)
marca su licencia como **`unknown`**, no como una licencia abierta específica.
La [página del corpus original de Antonio Gulli](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
invita al uso académico no comercial, solicita citar la fuente y prohíbe
modificar o redistribuir el corpus bajo un nombre diferente; recuerda que
los derechos de cada noticia pertenecen a sus fuentes originales. La
versión de clasificación de cuatro clases fue construida posteriormente
por Zhang, Zhao y LeCun. **No se redistribuyen textos del corpus aquí**;
para cualquier otra reutilización hay que comprobar derechos y condiciones
de las fuentes correspondientes. Los archivos descargados localmente en
`data/` están excluidos de Git.

La semilla de los scripts está en `src/config.py`; el conjunto de prueba no
debe utilizarse para escoger modelos ni hiperparámetros.
