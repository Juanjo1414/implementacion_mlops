# Todas las etapas usan Python 3.12 y las versiones fijadas en uv.lock.
FROM python:3.12-slim AS dependencies

# Tomar uv de su imagen oficial evita instalar paquetes con pip.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

# Reutilizar el Python de la imagen y copiar archivos al entorno virtual.
ENV UV_PYTHON_DOWNLOADS=0 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./

# API y MLflow necesitan solo las dependencias principales.
RUN uv sync --locked --no-default-groups

# El cuaderno necesita PyCaret y las librerías de análisis exploratorio.
FROM dependencies AS notebook-dependencies
RUN uv sync --locked --no-default-groups --group eda --group pycaret

# MLflow: el volumen /mlflow conserva SQLite y los artefactos entre reinicios.
FROM python:3.12-slim AS mlflow
WORKDIR /app
COPY --from=dependencies /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 5050
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5050", \
     "--allowed-hosts", "mlflow:5050,localhost:5050,127.0.0.1:5050", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts"]

# Notebook: instala grupos opcionales con uv y comparte el directorio de modelos.
FROM python:3.12-slim AS notebook
WORKDIR /app
COPY --from=notebook-dependencies /app/.venv /app/.venv
COPY eda.ipynb pycaret_experimentos.ipynb pycaret_features.py 1553768847-housing.csv ./
ENV PATH="/app/.venv/bin:$PATH" MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# API (última etapa: es la que construyen Render y Railway con este Dockerfile).
# El modelo, los datos subidos y el historial viven en el volumen DATA_DIR; los
# cron jobs usan esta misma imagen con otro comando (python cron_job.py ...).
FROM python:3.12-slim AS api
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=dependencies /app/.venv /app/.venv
COPY pycaret_app.py pycaret_features.py model_store.py cron_job.py 1553768847-housing.csv ./
COPY static ./static
# PORT lo reemplazan Render y Railway; SEED_MODEL_PATH solo existe si se monta models/.
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PORT=8000 DATA_DIR=/data \
    SEED_MODEL_PATH=/app/models/pycaret_housing.pkl
EXPOSE 8000
# Sin modelo previo, el arranque entrena uno inicial: start-period le da tiempo.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1
CMD ["sh", "-c", "exec uvicorn pycaret_app:app --host 0.0.0.0 --port ${PORT}"]
