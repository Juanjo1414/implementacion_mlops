"""API FastAPI para el pipeline exportado por pycaret_experimentos.ipynb."""

import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pycaret_features import OCEAN_CODES, prepare_pycaret_features


# La ruta es compartida por notebook, Docker Compose y servicio local.
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/pycaret_housing.pkl"))
logger = logging.getLogger(__name__)

# Cargar una vez evita abrir el archivo en cada solicitud.
try:
    model = joblib.load(MODEL_PATH)
except (FileNotFoundError, OSError, ValueError) as error:
    model = None
    logger.warning("No se pudo cargar %s: %s", MODEL_PATH, error)

app = FastAPI(title="California Housing · PyCaret", version="1.0.0")


class HousingData(BaseModel):
    """Variables originales de una vivienda; coinciden con el CSV."""

    longitude: float
    latitude: float
    housing_median_age: float = Field(ge=0)
    total_rooms: float = Field(ge=0)
    total_bedrooms: float | None = Field(default=None, ge=0)
    population: float = Field(ge=0)
    households: float = Field(ge=0)
    median_income: float = Field(ge=0)
    ocean_proximity: str


@app.get("/health")
def health() -> dict:
    """Responde 503 hasta que exista un modelo listo para inferencia."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo PyCaret no disponible")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(data: HousingData) -> dict:
    """Valida la categoría, transforma una fila y devuelve su predicción."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo PyCaret no disponible")

    # Validar antes de transformar evita devolver un error interno por categoría.
    if data.ocean_proximity not in OCEAN_CODES:
        raise HTTPException(status_code=422, detail="ocean_proximity no válida")

    # Se utiliza la misma función de esquema que el notebook de entrenamiento.
    frame = prepare_pycaret_features(pd.DataFrame([data.model_dump()]))

    # sklearn devuelve un arreglo incluso cuando solo llega una vivienda.
    value = float(model.predict(frame)[0])
    return {"predicted_median_house_value": value}
