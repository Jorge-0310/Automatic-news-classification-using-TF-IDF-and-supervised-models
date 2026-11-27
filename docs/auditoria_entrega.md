# Auditoría de entrega (estado verificable)

## Cerrado o parcialmente verificado

- [x] `docs/AGENTS.md` excluido de Git (`git check-ignore -v docs/AGENTS.md`).
- [x] `requirements.txt` fija `pyarrow==24.0.0` junto a NumPy 1.26.4,
  `datasets==5.2.0` y Streamlit 1.28.1.
- [x] Instalación en `venv` nuevo de Windows/Python 3.11: `pip check`
  no detectó conflictos; se importaron las cinco bibliotecas.
- [x] Tres pruebas unitarias pasan en el entorno nuevo, incluida la prueba
  de que el resumen de errores no contiene textos.
- [x] Cinco entradas del prototipo en entorno de desarrollo: tres temas,
  texto vacío y texto ambiguo (`python tests/smoke_app.py`).
- [x] Carpeta `notebooks/` con cuaderno exploratorio sin salidas con textos;
  sus tres celdas de código se ejecutaron sin errores en el entorno nuevo
  (30 000 ejemplos por clase y 37,85 palabras de media en entrenamiento).
- [x] Licencia de la ficha HF consultada (`unknown`) y condiciones del
  corpus original documentadas en `data/README.md`.
- [x] Entrenamiento completo **en entorno nuevo y copia temporal**:
  `logistic_bigrams` fue seleccionado de nuevo, accuracy 0,9193 y F1 macro
  0,9192 en prueba (comprobado en el `run_config.json` de esa copia).
- [x] Prototipo en entorno nuevo con modelo recién generado:
  `python tests/smoke_app.py` verificó cuatro textos con probabilidades
  y la respuesta a texto vacío. La primera ejecución del test agotó el
  tiempo predeterminado de AppTest (3 s) al cargar el modelo; se elevó
  a 90 s y el mismo flujo pasó.

## Pendientes humanos o de publicación

- [ ] Un integrante lee y confirma/corrige al menos diez errores con
  `python src/review_errors.py` (el resultado no debe incluir textos).
- [ ] Reemplazar en el informe los borradores IA por recuentos y
  observaciones humanas realmente revisadas.
- [ ] Confirmar contribuciones de Jorge y Anderson y completar verificaciones,
  reflexión y declaración de uso de IA en la bitácora.
- [ ] Dar formato IEEE real de dos columnas, comprobar 6–8 páginas,
  revisar referencias, corregir el texto y exportar PDF.
- [ ] Después de registrar los cambios y resolver los pendientes, poner
  la versión final en la rama predeterminada `main` y verificar el enlace
  público. No se hizo un merge/push automático desde un árbol de trabajo
  que contiene un documento eliminado (`docs/Plan de acción...md`); debe
  aclararse si esa eliminación es intencional.
- [ ] Clonar la revisión final y repetir los comandos del README en el
  sistema de destino; el chequeo limpio actual solo cubre Windows/Python 3.11.

Los artefactos versionables de resultados están en `experiments/`; el modelo
entrenado y el CSV con textos se mantienen fuera de Git.
