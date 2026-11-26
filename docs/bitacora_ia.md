# Bitácora de uso de IA

Las tres primeras entradas provienen de la bitácora previa; no constan en ella
el integrante concreto ni una verificación documentada. No se atribuyen
retroactivamente verificaciones que no estén registradas. Quedan para completar
por quienes hicieron esas consultas.

| Fecha | Integrante | Herramienta | Propósito / solicitud registrada | Aceptado, modificado o descartado | Verificación |
|---|---|---|---|---|---|
| 14/11/2026 | Por confirmar | Manus | Dar formato a `docs/avance_hito_2.md` y revisar cumplimiento del hito. | Se obtuvo estructura para completar con resultados. No consta detalle de modificaciones. | No registrada en la entrada original. |
| 15/11/2026 | Por confirmar | Manus | Generar código para actualizar automáticamente `results/` tras ejecutar la línea base. | Se integró un bloque que escribe resultados; no consta detalle de modificaciones. | No registrada en la entrada original. |
| 16/11/2026 | Por confirmar | Claude | Generar un `.docx` unificando bitácora e Hito 2. | Se obtuvo un documento de resumen; no consta detalle de modificaciones. | No registrada en la entrada original. |
| 26/11/2026 | Solicitante del repositorio (identidad por confirmar) | OpenCode / GPT-6 Sol | Revisar plan y completar estructura reproducible; posteriormente implementar la comparación de tres modelos y avanzar con el resto del plan. | Se aceptó código tras adaptar la selección para usar exclusivamente validación; se descartó presentar las métricas anteriores de prueba como selección justa. Se corrigió un error real del extractor: duplicaba índices por captura de variable en un generador. | Compilación Python y prueba unitaria de cuatro pipelines; ejecución real de `src/train.py` con F1 macro 0,9192 en prueba, respaldada por `experiments/run_config.json`; inferencia del modelo guardado y prueba de la app con `AppTest`; comprobación de 20 índices de error únicos. Las 12 hipótesis sobre errores aún requieren revisión humana. |

## Reflexión pendiente de validación por el equipo

La línea base histórica se midió sobre el conjunto de prueba antes de definir
la comparación. Utilizar esa cifra para elegir configuraciones daría una
ventaja indebida y haría que la prueba dejara de ser una evaluación final
independiente. Por eso el nuevo script selecciona solo con validación y
reserva prueba para el ganador. Esta precaución técnica no demuestra por sí
sola que la ejecución final sea correcta: hay que verificar los resultados
generados y revisar las cifras antes de incluirlas en el informe. El equipo
debe completar su propia reflexión sobre propuestas de IA concretas que haya
rechazado o corregido, con evidencia de la revisión.
