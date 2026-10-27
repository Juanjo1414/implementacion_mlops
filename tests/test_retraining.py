"""Reentrenamiento con datos subidos, rutas de administración y cron jobs.

Cada prueba usa el fixture `store` (tests/conftest.py): un ModelStore en un
directorio temporal, con 500 filas base y un RandomForest de 10 árboles, para
que todo el archivo corra en un par de segundos sin tocar ./runtime.
"""

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import cron_job
import pycaret_app
from model_store import BatchError, validate_batch


# Valor de ADMIN_TOKEN que las pruebas ponen en el entorno.
TOKEN = "token-de-prueba"


def labeled_batch(rows: int = 50, seed: int = 1) -> pd.DataFrame:
    """Filas reales del CSV original, como si fueran un lote nuevo etiquetado."""
    return pd.read_csv("1553768847-housing.csv").sample(rows, random_state=seed)


def as_csv(frame: pd.DataFrame) -> bytes:
    """Bytes de un CSV, como los que envía la interfaz al subir un archivo."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


@pytest.fixture
def client(store, monkeypatch):
    """Cliente HTTP con token configurado y el modelo inicial ya entrenado.

    Sin `with TestClient(...)` no se ejecuta el lifespan de la API, así que el
    modelo se prepara aquí de forma síncrona.
    """
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)
    store.load_or_bootstrap()
    return TestClient(pycaret_app.app)


def test_bootstrap_trains_and_records_initial_champion(store):
    """Sin modelo previo ni .pkl del cuaderno se entrena el campeón v1."""
    store.load_or_bootstrap()
    state = store.read_state()
    assert store.model is not None and store.version == "v1"
    assert state["champion"]["trigger"] == "inicial"
    assert state["champion"]["algorithm"] == "RandomForestRegressor"
    assert store.champion_path.exists()


def test_restart_loads_saved_champion_without_retraining(store, base_csv):
    """Tras un reinicio se carga champion.pkl; pipeline_factory=None prueba que no se entrena."""
    store.load_or_bootstrap()
    restarted = type(store)(store.data_dir, base_csv, pipeline_factory=None)
    restarted.load_or_bootstrap()
    assert restarted.version == "v1"
    assert len(restarted.read_state()["history"]) == 1


def test_validate_batch_reports_every_problem():
    """Un lote con varios errores los informa todos, con su número de línea."""
    # Como object, la columna admite texto igual que un CSV mal escrito.
    frame = labeled_batch(20).astype({"median_income": object})
    frame.iloc[0, frame.columns.get_loc("median_income")] = "abc"
    frame.iloc[1, frame.columns.get_loc("ocean_proximity")] = "MARS"
    frame.iloc[2, frame.columns.get_loc("population")] = -5
    with pytest.raises(BatchError) as error:
        validate_batch(frame)
    messages = " ".join(error.value.errors)
    assert "median_income" in messages and "líneas 2" in messages
    assert "MARS" in messages
    assert "population" in messages

    with pytest.raises(BatchError, match="Faltan columnas: median_house_value"):
        validate_batch(frame.drop(columns="median_house_value"))


def test_validate_batch_allows_missing_bedrooms():
    """total_bedrooms es la única columna que puede venir vacía."""
    frame = labeled_batch(20)
    frame["total_bedrooms"] = None
    assert validate_batch(frame)["total_bedrooms"].isna().all()


def test_retrain_without_new_data_requests_it(store):
    """Sin lotes nuevos no se entrena: se marca la solicitud y subir un lote la quita."""
    store.load_or_bootstrap()
    result = store.retrain("cron")
    assert result["status"] == "awaiting_data"
    assert store.read_state()["data_requested_at"] == result["finished_at"]

    store.add_batch(labeled_batch())
    assert store.read_state()["data_requested_at"] is None


def test_retrain_uses_uploaded_batch_and_compares_on_same_holdout(store):
    """El retador entrena con el 80 % del lote y ambos se evalúan con el 20 % restante."""
    store.load_or_bootstrap()
    batch = store.add_batch(labeled_batch(50))

    result = store.retrain("cron")

    assert result["status"] in {"promoted", "kept"}
    assert result["new_batches"] == [batch["id"]]
    # 500 filas base + 50 nuevas, partidas 80/20 cada una.
    assert result["train_rows"] == 400 + 40
    assert result["eval_rows"] == 100 + 10
    assert store.pending_batches() == []
    if result["status"] == "promoted":
        assert store.version == "v2" == store.read_state()["champion"]["version"]


def test_admin_routes_require_token(client):
    """Sin token o con uno incorrecto, subir datos y reentrenar responden 401."""
    assert client.post("/retrain").status_code == 401
    response = client.post("/data/upload", content=b"x", headers={"X-Admin-Token": "otro"})
    assert response.status_code == 401


def test_upload_and_retrain_through_api(client, store):
    """Flujo completo por HTTP: CSV inválido → 422, válido → 201, reentrenar → 202."""
    headers = {"X-Admin-Token": TOKEN, "Content-Type": "text/csv"}

    bad = client.post("/data/upload", content=as_csv(labeled_batch().drop(columns="households")), headers=headers)
    assert bad.status_code == 422
    assert "households" in bad.json()["detail"][0]

    uploaded = client.post("/data/upload", content=as_csv(labeled_batch()), headers=headers)
    assert uploaded.status_code == 201
    assert client.get("/status").json()["pending_rows"] == 50

    started = client.post("/retrain?trigger=cron", headers=headers)
    assert started.status_code == 202
    store._thread.join(timeout=60)
    assert client.get("/retrain/status").json()["status"] in {"promoted", "kept"}

    # Sin lotes pendientes, el siguiente reentrenamiento pide datos.
    assert client.post("/retrain", headers=headers).json()["status"] == "awaiting_data"


def test_predictions_are_logged_except_health_checks(client):
    """Las predicciones reales se registran; las del cron de salud no."""
    house = cron_job.SAMPLE_HOUSE
    assert client.post("/predict", json=house).status_code == 200
    assert client.post("/predict", json=house, headers={"X-Health-Check": "true"}).status_code == 200
    extreme = {**house, "population": 1_000_000}
    assert client.post("/predict", json=extreme).json()["outliers"][0]["feature"] == "population"

    summary = client.get("/monitor/summary").json()
    assert summary["total_predictions"] == 2
    assert summary["outlier_rate"] == 0.5


def test_interface_is_served(client):
    """La raíz de la API devuelve la interfaz HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Datos nuevos y reentrenamiento" in response.text


def fake_api(responses: dict):
    """Sustituye cron_job.call por respuestas fijas según (método, ruta)."""
    def call(method, path, body=None, headers=None):
        return responses[(method, path.split("?")[0])]
    return call


def test_cron_health_fails_when_model_is_down(monkeypatch):
    """El cron de salud sale con 1 si /health falla y con 0 si todo responde bien."""
    monkeypatch.setattr(cron_job, "call", fake_api({("GET", "/health"): (503, {"detail": "sin modelo"})}))
    assert cron_job.check_health() == 1

    monkeypatch.setattr(cron_job, "call", fake_api({
        ("GET", "/health"): (200, {"status": "healthy"}),
        ("POST", "/predict"): (200, {"predicted_median_house_value": 412000.0, "model_version": "v1"}),
    }))
    assert cron_job.check_health() == 0


def test_cron_retrain_waits_for_result(monkeypatch):
    """El cron espera mientras el estado sea running; falta de datos no es error, token malo sí."""
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(cron_job.time, "sleep", lambda seconds: None)
    statuses = iter([(200, {"status": "running"}), (200, {"status": "kept", "champion_metrics": {"rmse": 1}})])
    monkeypatch.setattr(cron_job, "call", lambda method, path, body=None, headers=None: (
        (202, {"status": "started"}) if method == "POST" else next(statuses)
    ))
    assert cron_job.request_retrain() == 0

    monkeypatch.setattr(cron_job, "call", fake_api({("POST", "/retrain"): (200, {"status": "awaiting_data"})}))
    assert cron_job.request_retrain() == 0

    monkeypatch.setattr(cron_job, "call", fake_api({("POST", "/retrain"): (401, {"detail": "token"})}))
    assert cron_job.request_retrain() == 1


def test_cron_end_to_end_against_running_api(client, monkeypatch):
    """Los cron jobs funcionan contra la API real (TestClient en lugar de urllib)."""
    def call(method, path, body=None, headers=None):
        response = client.request(method, path, json=body, headers=headers)
        return response.status_code, response.json()

    monkeypatch.setattr(cron_job, "call", call)
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)
    assert cron_job.check_health() == 0
    assert cron_job.request_retrain() == 0
    assert pycaret_app.store.read_state()["data_requested_at"] is not None
