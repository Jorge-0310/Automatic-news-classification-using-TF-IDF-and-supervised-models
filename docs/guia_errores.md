# Guía de análisis de errores (revisión manual)

Después de ejecutar `python src/train.py`, ejecute
`python src/error_analysis.py`. El archivo `experiments/errors.csv` se crea
solo en su equipo, se ignora en Git y contiene texto del corpus: **no lo
publique** sin verificar permisos de las fuentes.

Revise al menos diez filas. Para cada una, llene manualmente `error_type`,
`cause_hypothesis` y `limitation`, sin inferir una causa solo a partir de
la clase predicha o de la probabilidad. Las siguientes son hipótesis de
categorías posibles, **no etiquetas asignadas automáticamente**:

Hay 12 borradores de categorización en `experiments/error_review_draft.json`.
Puede aplicarlos al archivo local con `python src/annotate_errors.py`;
su origen es IA y **debe revisar y corregir** cada uno antes de considerarlos
un análisis manual. El script de extracción no sobreescribirá un archivo
ya anotado.

| Tipo posible | Hipótesis que debe contrastarse | Límite que ilustra |
|---|---|---|
| Business / Sci-Tech | Aparecen términos compartidos sobre tecnología y empresas. | Una sola etiqueta omite la superposición temática. |
| World / Sports o Business | Un evento internacional menciona competición o mercados. | Las palabras clave no sustituyen la lectura contextual. |
| Texto corto o ambiguo | Faltan pistas suficientes en el fragmento recibido. | TF-IDF no recupera contexto externo. |
| Múltiples temas | El artículo cubre más de una categoría. | AG News impone una etiqueta por noticia. |
| Nombres o siglas | Una entidad no aparece o aparece poco en entrenamiento. | El vocabulario aprendido no generaliza a toda entidad nueva. |
| Contexto implícito | El tema depende de información que no figura en el texto. | Ningún clasificador puede observar el contexto ausente. |

Al escribir el informe, agrupe los casos realmente observados por tipo,
cuente cuántos se revisaron y describa al menos una limitación concreta
por grupo. Si ninguna categoría propuesta aplica, cree una nueva y
explique por qué. No sustituya este análisis por ejemplos inventados.
