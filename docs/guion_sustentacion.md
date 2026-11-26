# Guion breve de sustentación — cifras verificadas

## Resumen de dos minutos

NewsScope ayuda a ordenar noticias escritas en inglés entre cuatro categorías:
World, Sports, Business y Sci/Tech. No verifica la veracidad. Usamos AG News,
con 120 000 ejemplos de entrenamiento y 7 600 de prueba. Un pipeline ajusta
TF-IDF y un clasificador; comparamos regresión logística con unigramas,
regresión logística con bigramas, ComplementNB y SGDClassifier con bigramas.
Separamos una validación estratificada de 20 % del entrenamiento con semilla
42. Se escoge la configuración con mejor F1 macro en validación; la prueba
se reserva para medir únicamente la ganadora tras reentrenarla con todos
los ejemplos de entrenamiento. Ganó regresión logística con bigramas:
F1 macro 0,9192 en validación y 0,9192 en prueba, con accuracy 0,9193
en prueba. ComplementNB (0,9138) y SGD (0,8922) quedaron por debajo
en validación. Los límites principales son el
idioma del corpus, su fecha y las confusiones entre temas; el prototipo
requiere revisión humana.

## Preguntas previsibles

- **¿Por qué F1 macro?** Da igual peso a las cuatro clases; accuracy puede
  ocultar una clase débil incluso con rendimiento global alto.
- **¿Por qué TF-IDF dentro del pipeline?** Para que vocabulario e IDF se
  aprendan solo con la porción de entrenamiento en cada evaluación.
- **¿Qué diferencia a los modelos?** Logística aprende probabilidades
  discriminativas; ComplementNB usa conteos complementarios; SGD aprende
  pesos lineales mediante actualizaciones estocásticas con pérdida logarítmica.
- **¿Qué hacen los bigramas?** Incorporan secuencias de dos términos;
  aumentan las características y pueden capturar contexto o sobreajustarse.
- **¿Por qué prueba una sola vez?** Consultarla para elegir parámetros
  convertiría la medición final en parte del ajuste.
- **¿Qué significa una matriz de confusión?** Filas de clases reales y
  columnas de predicciones; fuera de la diagonal aparecen los errores.
- **¿Efecto de `C` y `alpha`?** `C` mayor reduce la regularización de
  logística; `alpha` en ComplementNB suaviza conteos; `alpha` en SGD
  controla la penalización, por lo que no significa lo mismo en ambos.
- **¿Cómo se revisó la IA?** No se tomó el resultado de prueba de la línea
  base como criterio de selección comparativa. Se comprobó que el código
  separa validación antes de ajustar TF-IDF y que el ganador se decide antes
  de evaluar prueba. Consulte bitácora y resultados; no afirmar
  verificaciones o análisis manual aún no realizados.
