"""Comprueba el contrato entre columnas del notebook y la API PyCaret."""

import pandas as pd
from fastapi.testclient import TestClient

import pycaret_app
from pycaret_features import PYCARET_FEATURES, prepare_pycaret_features


SAMPLE_HOUSE = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": None,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}


def test_prepared_features_use_fixed_schema():
    """La categoría y el valor faltante se transforman igual para train/serve."""
    result = prepare_pycaret_features(pd.DataFrame([SAMPLE_HOUSE]))
    assert result.columns.tolist() == PYCARET_FEATURES
    assert result.loc[0, "ocean_proximity_code"] == 3
    assert result.loc[0, "bedrooms_missing"] == 1
    assert result.loc[0, "total_bedrooms"] == -1


def test_pycaret_api_predicts_with_prepared_schema(store, monkeypatch):
    """La API envía al pipeline exactamente las columnas exportadas."""
    class RecordingModel:
        def predict(self, frame):
            assert frame.columns.tolist() == PYCARET_FEATURES
            return [250000.0]

    monkeypatch.setattr(store, "model", RecordingModel())
    client = TestClient(pycaret_app.app)
    assert client.get("/health").status_code == 200
    response = client.post("/predict", json=SAMPLE_HOUSE)
    assert response.status_code == 200
    assert response.json()["predicted_median_house_value"] == 250000.0


def test_pycaret_api_rejects_unknown_category_and_missing_model(store, monkeypatch):
    """Errores de entrada y falta del artefacto tienen códigos HTTP claros."""
    monkeypatch.setattr(store, "model", object())
    client = TestClient(pycaret_app.app)
    bad_house = {**SAMPLE_HOUSE, "ocean_proximity": "UNKNOWN"}
    assert client.post("/predict", json=bad_house).status_code == 422

    monkeypatch.setattr(store, "model", None)
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=SAMPLE_HOUSE).status_code == 503
