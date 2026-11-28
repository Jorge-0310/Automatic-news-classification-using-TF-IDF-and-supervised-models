# IA responsable y límites de NewsScope

NewsScope solo asigna una etiqueta temática entre cuatro clases a noticias
en inglés. No detecta falsedades, intención, importancia ni calidad editorial.
AG News puede reflejar sesgos geográficos, editoriales y temporales; las cuatro
categorías no son una taxonomía universal. No se puede extrapolar el desempeño
sin evaluación nueva a noticias actuales, locales o en español.

La línea base preliminar reportó peor F1 en `Business` y `Sci/Tech` que en
`Sports`. Las noticias sobre tecnología empresarial y las noticias con varios
temas son fuentes plausibles de confusión, pero hay que comprobarlas en errores
reales antes de atribuirles una causa. Una predicción o probabilidad elevada no
demuestra que la noticia sea verdadera.

**Usos excluidos:** moderación automática, selección o priorización de personas,
decisiones financieras y verificación de noticias. No introducir textos con
datos personales en la aplicación sin autorización. La aplicación se ejecuta
localmente y no registra entradas deliberadamente, pero el usuario sigue siendo
responsable del tratamiento de los datos.

**Mitigaciones:** revisión humana antes de actuar, advertencia visible en la
interfaz, métricas desagregadas por clase, análisis manual de errores, registro
de versiones/semillas, corpus no redistribuido y reevaluación con datos locales
autorizados antes de cualquier despliegue. El modelo serializado debe cargarse
solo si fue generado en un entorno de confianza (joblib puede ejecutar código).
