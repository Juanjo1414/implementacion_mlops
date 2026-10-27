"""API FastAPI del pipeline PyCaret: interfaz, monitoreo y reentrenamiento.

El modelo, los datos subidos y el historial viven en DATA_DIR (model_store.py).
Este archivo solo traduce HTTP a llamadas del ModelStore y define qué rutas
exigen el encabezado X-Admin-Token.

Rutas públicas:  /  /health  /predict  /status  /monitor/summary  /retrain/status
Rutas con token: /data/upload  /retrain

Ejecutar en local:
    ADMIN_TOKEN=secreto uv run uvicorn pycaret_app:app --reload
"""

import io
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from model_store import BatchError, ModelStore
from pycaret_features import OCEAN_CODES, prepare_pycaret_features


# Render y Railway muestran la salida estándar como logs del servicio; sin esta
# configuración los mensajes INFO de ModelStore no aparecerían.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# La interfaz es un único HTML estático junto a este archivo.
STATIC_DIR = Path(__file__).parent / "static"
# Límite de tamaño de los CSV: protege la memoria del contenedor.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Una sola instancia para todo el proceso. Las pruebas la reemplazan por una
# que usa un directorio temporal (tests/conftest.py).
store = ModelStore.from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Se ejecuta una vez al arrancar el servidor, antes de atender solicitudes.

    Entrenar el modelo inicial puede tardar; en un hilo, el servidor empieza a
    responder de inmediato y /health devuelve 503 hasta que el modelo esté
    listo. Render y Railway esperan ese 200 antes de enviar tráfico.
    """
    threading.Thread(target=store.load_or_bootstrap, name="bootstrap", daemon=True).start()
    yield


app = FastAPI(title="California Housing · PyCaret", version="2.0.0", lifespan=lifespan)


class HousingData(BaseModel):
    """Variables originales de una vivienda; coinciden con el CSV.

    Pydantic rechaza con 422 los tipos incorrectos y los valores negativos
    antes de que lleguen al modelo.
    """

    longitude: float
    latitude: float
    housing_median_age: float = Field(ge=0)
    total_rooms: float = Field(ge=0)
    total_bedrooms: float | None = Field(default=None, ge=0)
    population: float = Field(ge=0)
    households: float = Field(ge=0)
    median_income: float = Field(ge=0)
    ocean_proximity: str


def require_admin(x_admin_token: str | None = Header(default=None)):
    """Dependencia de FastAPI para las rutas que cambian datos o modelo.

    FastAPI convierte el parámetro `x_admin_token` en el encabezado
    `X-Admin-Token`. Sin ADMIN_TOKEN configurado la administración queda
    deshabilitada (es más seguro que dejarla abierta por olvido).
    """
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Configure ADMIN_TOKEN para habilitar la administración")
    # compare_digest tarda lo mismo acierte o no, así que no revela cuántos
    # caracteres del token coinciden (ataque de tiempo).
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Token de administración inválido")


@app.get("/", include_in_schema=False)
def interface():
    """Interfaz web; habla con esta misma API desde el navegador."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Responde 503 hasta que exista un modelo listo para inferencia.

    La usan el health check de Render/Railway, el HEALTHCHECK de Docker y el
    cron `python cron_job.py health`.
    """
    if store.model is None:
        raise HTTPException(status_code=503, detail="Modelo PyCaret no disponible")
    return {"status": "healthy", "model_loaded": True, "model_version": store.version}


@app.post("/predict")
def predict(data: HousingData, x_health_check: bool = Header(default=False)) -> dict:
    """Valida la categoría, transforma una fila y devuelve su predicción.

    El encabezado opcional `X-Health-Check: true` lo envía el cron de salud
    para que su predicción de prueba no cuente en las estadísticas.
    """
    # Copia local: un reentrenamiento puede reemplazar store.model en paralelo.
    model = store.model
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo PyCaret no disponible")

    # Validar antes de transformar evita devolver un error interno por categoría.
    if data.ocean_proximity not in OCEAN_CODES:
        raise HTTPException(status_code=422, detail="ocean_proximity no válida")

    # Se utiliza la misma función de esquema que el notebook de entrenamiento.
    features = data.model_dump()
    frame = prepare_pycaret_features(pd.DataFrame([features]))

    # sklearn devuelve un arreglo incluso cuando solo llega una vivienda.
    value = float(model.predict(frame)[0])

    # Las pruebas del cron de salud no cuentan como tráfico real.
    outliers = store.outliers(features) if x_health_check else store.log_prediction(features, value)
    return {"predicted_median_house_value": value, "model_version": store.version, "outliers": outliers}


@app.get("/status")
def model_status() -> dict:
    """Campeón, datos pendientes e historial de reentrenamientos."""
    return store.status()


@app.get("/monitor/summary")
def monitor_summary() -> dict:
    """Resumen de las predicciones guardadas en el volumen."""
    return store.prediction_summary()


@app.post(
    "/data/upload",
    status_code=201,
    dependencies=[Depends(require_admin)],
    # Documenta en Swagger (/docs) que el cuerpo es un CSV en texto plano.
    openapi_extra={"requestBody": {"required": True, "content": {"text/csv": {"schema": {"type": "string"}}}}},
)
async def upload_data(request: Request) -> dict:
    """Recibe un CSV etiquetado (cuerpo text/csv) para el próximo reentrenamiento.

    Se lee el cuerpo crudo en vez de usar un formulario multipart: así no hace
    falta la dependencia python-multipart y funciona igual desde curl
    (`--data-binary @archivo.csv`) y desde la interfaz.

        curl -X POST https://<api>/data/upload -H "X-Admin-Token: …" \\
             -H "Content-Type: text/csv" --data-binary @nuevos.csv
    """
    body = await request.body()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="El CSV supera 20 MB")
    try:
        frame = pd.read_csv(io.BytesIO(body))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=422, detail=[f"No se pudo leer el CSV: {error}"])
    try:
        # Validar y escribir en disco bloquea; en un hilo aparte el servidor
        # sigue atendiendo otras solicitudes mientras tanto.
        return await run_in_threadpool(store.add_batch, frame)
    except BatchError as error:
        raise HTTPException(status_code=422, detail=error.errors)


# Traducción del resultado de ModelStore.start_retrain a código HTTP. El cron
# lee estos códigos para decidir qué escribir en los logs.
RETRAIN_CODES = {"started": 202, "running": 409, "awaiting_data": 200, "not_ready": 503}


@app.post("/retrain", dependencies=[Depends(require_admin)])
def retrain(trigger: Literal["manual", "cron"] = "manual") -> JSONResponse:
    """Entrena un retador en segundo plano si hay datos nuevos; si no, los pide.

    Responde enseguida (202) porque entrenar puede tardar más que el tiempo
    máximo de una petición HTTP; el progreso se consulta en /retrain/status.
    `trigger` solo queda anotado en el historial (cron = programado).
    """
    result = store.start_retrain(trigger)
    return JSONResponse(result, status_code=RETRAIN_CODES[result["status"]])


@app.get("/retrain/status")
def retrain_status() -> dict:
    """Entrenamiento en curso (`running`) o resultado del último (`promoted`, `kept`…)."""
    return store.retrain_status()
