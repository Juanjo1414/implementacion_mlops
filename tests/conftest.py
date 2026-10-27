"""Fixtures compartidas: un ModelStore aislado en un directorio temporal."""

import pandas as pd
import pytest

import pycaret_app
from model_store import ModelStore, default_pipeline


def small_pipeline():
    """Mismo pipeline que producción con pocos árboles para que las pruebas sean rápidas."""
    return default_pipeline().set_params(rf__n_estimators=10, rf__n_jobs=1)


@pytest.fixture
def base_csv(tmp_path):
    path = tmp_path / "base.csv"
    pd.read_csv("1553768847-housing.csv").sample(500, random_state=0).to_csv(path, index=False)
    return path


@pytest.fixture
def store(tmp_path, base_csv, monkeypatch):
    """ModelStore sin modelo cargado, usado también por la API durante la prueba."""
    model_store = ModelStore(
        data_dir=tmp_path / "data",
        base_data_path=base_csv,
        seed_model_path=None,
        pipeline_factory=small_pipeline,
    )
    monkeypatch.setattr(pycaret_app, "store", model_store)
    return model_store
