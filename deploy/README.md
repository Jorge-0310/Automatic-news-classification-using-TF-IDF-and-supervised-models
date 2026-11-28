# Desplegar NewsScope desde el repositorio público

La app pública es `deploy/streamlit_app.py`; Community Cloud instala
`deploy/requirements.txt`. **No se sube `models/newsscope.joblib` a GitHub**:
continúa excluido por `.gitignore`. Se almacena en un repositorio **privado**
de modelos de Hugging Face Hub y la app lo descarga al primer uso. El SHA-256
esperado está en `deploy/model.sha256`; la app comprueba el hash antes de
deserializar el joblib. No se descarga ni redistribuye el corpus de AG News.

## Paso a paso

1. **Revisar derechos de uso.** Lea `data/README.md`: la licencia publicada de
   AG News figura como *unknown* y las noticias conservan derechos de sus
   autores. Resuelva si la demo pública y el almacenamiento privado del modelo
   derivado son compatibles con las condiciones aplicables; el código no
   concede licencias.
2. **Comprobar el modelo local** (si falta, ejecutar `python src/train.py`).
   Verifique que el hash del artefacto coincide con `deploy/model.sha256`:

   ```powershell
   (Get-FileHash models/newsscope.joblib -Algorithm SHA256).Hash.ToLower()
   Get-Content deploy/model.sha256
   ```

   En Linux/macOS: `sha256sum models/newsscope.joblib`.
   **No actualice el hash** salvo que haya entrenado y revisado personalmente
   un modelo nuevo. Nunca cargue modelos joblib ajenos: pueden ejecutar código.
3. **Alojar el modelo en privado.** Cree una cuenta en
   [Hugging Face](https://huggingface.co/join) y un repositorio nuevo de tipo
   **Model**, visibilidad **Private** (p. ej. `usuario/newsscope-private`).
   Desde su página web, suba únicamente el archivo local
   `models/newsscope.joblib` a la raíz con el nombre `newsscope.joblib`.
   Compruebe que la visibilidad sigue siendo Private. No suba corpus ni
   `experiments/errors.csv`. El host privado puede custodiar el modelo, pero
   no lo vuelve seguro frente a personas con permiso de escritura.
4. **Crear una credencial de solo lectura.** En [Access Tokens de Hugging
   Face](https://huggingface.co/settings/tokens), cree un token *fine-grained*
   con permiso de **lectura** solo para ese repo privado (o el mínimo acceso
   que permita el proveedor). Copie su valor y no lo agregue a Git.
5. **Publicar los cambios de código** en la rama de GitHub que usará
   Streamlit. Confirme que `git status` no incluye el `.joblib`, secretos,
   corpus o textos de `errors.csv`. En particular deben estar disponibles
   `deploy/streamlit_app.py`, `deploy/editorial.css`, `deploy/requirements.txt`,
   `.streamlit/config.toml`, `deploy/model.sha256`,
   `experiments/dashboard_metrics.json` y
   `experiments/error_summary_public.json`.
6. En [Streamlit Community Cloud](https://share.streamlit.io/), conecte
   GitHub y elija **Create app → Yup, I have an app**. Seleccione el
   **repositorio público existente**, la rama publicada y el entrypoint
   **`deploy/streamlit_app.py`**. En Advanced settings elija Python **3.11**
   y configure **Secrets** (reemplace los valores, no incluya comillas
   adicionales):

   ```toml
   HF_MODEL_REPO = "usuario/newsscope-private"
   HF_TOKEN = "hf_token_de_solo_lectura"
   ```

   No suba `.streamlit/secrets.toml`; está ignorado. Community Cloud
   lee las dependencias del directorio de la app antes que las del
   repositorio raíz. El tema claro se configura en `.streamlit/config.toml`
   **en la raíz del repositorio** (la configuración del subdirectorio
   no se aplicó en este despliegue).
7. Pulse **Deploy**. Un repositorio fuente público produce una app pública
   por defecto. Abra la URL `https://...streamlit.app`, clasifique un ejemplo
   de cada categoría y revise las vistas Rendimiento y Errores. La primera
   predicción descarga ~44 MB del repo privado y puede tardar. Si falla,
   compruebe el nombre del archivo, token, permisos, hash y logs **sin
   compartir el token**. En nuevas versiones del modelo, actualice
   conscientemente archivo y SHA-256 y vuelva a probar.

Para probar localmente desde la **raíz** del proyecto:
`python -m pip install -r deploy/requirements.txt` y
`python -m streamlit run deploy/streamlit_app.py --server.address 127.0.0.1`.
El modelo local ignorado por Git tiene prioridad; no necesita Secrets.

## Límites de la publicación

Streamlit Community Cloud ofrece HTTPS. La app evita mostrar mensajes de
error del proveedor y no registra deliberadamente el texto introducido; las
entradas sí pasan por la infraestructura del servicio. Los límites en
memoria (5 por sesión y 30 globales por minuto/proceso) son **de cortesía**,
no protegen ante abuso distribuido ni ofrecen WAF configurable. Para un
servicio público con garantías fuertes, use un host con límites de red,
monitoreo y política de gastos. Community Cloud tiene recursos variables y
puede suspender instancias inactivas; no entrene el modelo en el arranque.

Documentación: [organización y dependencias de Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization),
[Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
y [repositorios privados en Hugging Face](https://huggingface.co/docs/hub/repositories-settings).
