"""Modelo campeón, lotes de datos subidos y reentrenamiento en el volumen DATA_DIR.

Este módulo es el "cerebro" de producción. La API (pycaret_app.py) solo traduce
HTTP a llamadas de ModelStore; los cron jobs (cron_job.py) solo llaman a la API.

Vocabulario
-----------
- **Campeón**: el modelo que está sirviendo predicciones ahora mismo.
- **Retador**: un modelo nuevo entrenado con más datos; reemplaza al campeón
  solo si comete menos error (RMSE) en datos que ninguno de los dos vio.
- **Lote**: un CSV etiquetado (con el valor real de cada vivienda) que sube un
  usuario. Queda "pendiente" hasta que un reentrenamiento lo usa.

Por qué todo vive en un solo directorio
---------------------------------------
En Render y Railway un disco persistente se conecta a un único servicio. Si los
cron jobs necesitaran leer el modelo tendríamos que montar un almacenamiento
compartido (S3, etc.). En cambio, todo el estado vive en el volumen de la API y
los cron jobs se comunican con ella por HTTP:

    DATA_DIR/
    ├── state.json           campeón, lotes subidos e historial de reentrenamientos
    ├── models/champion.pkl  pipeline que sirve la API
    ├── uploads/<lote>.csv   datos etiquetados subidos por los usuarios
    └── predictions.jsonl    registro persistente de predicciones (una por línea)
"""

import json
import logging
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pycaret_features import OCEAN_CODES, PYCARET_FEATURES, prepare_pycaret_features

logger = logging.getLogger("ModelStore")

# Columna que el modelo aprende a predecir (valor medio de la vivienda en USD).
TARGET = "median_house_value"

# Columnas originales del CSV: las mismas que recibe /predict. El pipeline no
# las usa tal cual; prepare_pycaret_features las convierte a PYCARET_FEATURES.
RAW_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
]
# Columnas que no pueden ser negativas; total_bedrooms además puede faltar.
NON_NEGATIVE = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

# Misma semilla y proporción que el cuaderno: la partición base coincide con
# data/train.csv y data/test.csv, así que el holdout nunca se usa para entrenar.
SEED = 42
EVAL_SIZE = 0.2
# Con menos de 10 filas, el 20 % de evaluación de un lote tendría 0 o 1 fila.
MIN_BATCH_ROWS = 10
# Mismo umbral que monitor.py: |valor - media| / desviación > 3 es atípico.
Z_SCORE_THRESHOLD = 3


class BatchError(ValueError):
    """El CSV subido no cumple el esquema; `errors` explica cada problema.

    La API la convierte en una respuesta 422 cuyo `detail` es esta lista, para
    que la interfaz muestre un mensaje por línea.
    """

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def utc_now() -> str:
    """Fecha y hora actual en UTC (ISO 8601); el navegador la muestra en hora local."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_pipeline() -> Pipeline:
    """Reproduce la estructura del pipeline que exporta pycaret_experimentos.ipynb.

    Solo se usa en el primer arranque sin modelo (en la nube, models/ no está en
    Git). Después, cada reentrenamiento clona el campeón vigente, así que el
    cuaderno sigue siendo el lugar donde se elige el algoritmo.

    Estructura (igual a la que imprime joblib.load('models/pycaret_housing.pkl')):
        preprocess: imputar faltantes con la media → estandarizar
        rf:         RandomForestRegressor con 100 árboles
    """
    numeric = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
    preprocess = ColumnTransformer(
        [("numerical_pipeline", numeric, PYCARET_FEATURES)],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            # n_jobs=-1 usa todos los núcleos del contenedor para entrenar más rápido.
            ("rf", RandomForestRegressor(n_jobs=-1, random_state=SEED)),
        ]
    )


def evaluate(model, frame: pd.DataFrame) -> dict:
    """Métricas de regresión del modelo sobre un conjunto etiquetado.

    - RMSE: error típico en dólares; penaliza más los errores grandes. Es la
      métrica que decide si el retador reemplaza al campeón.
    - MAE: error absoluto promedio en dólares; más fácil de explicar.
    - R²: proporción de la variación del precio que explica el modelo (1 = perfecto).
    """
    predictions = model.predict(prepare_pycaret_features(frame))
    y_true = frame[TARGET]
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, predictions))), 2),
        "mae": round(float(mean_absolute_error(y_true, predictions)), 2),
        "r2": round(float(r2_score(y_true, predictions)), 4),
    }


def _csv_lines(mask: pd.Series) -> str:
    """Números de línea del CSV (la 1 es el encabezado) para un mensaje de error.

    `mask` marca con True las filas con problema. Se muestran como máximo cinco
    líneas para que el mensaje no crezca sin límite con un archivo muy dañado.
    """
    # +2: las posiciones empiezan en 0 y la línea 1 del archivo es el encabezado.
    lines = np.flatnonzero(mask.to_numpy()) + 2
    shown = ", ".join(str(line) for line in lines[:5])
    return shown + ("…" if len(lines) > 5 else "")


def validate_batch(frame: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el lote con tipos numéricos o lanza BatchError con cada problema.

    Se rechaza el lote completo en vez de descartar filas en silencio: quien
    sube los datos debe saber qué corregir. Se acumulan todos los errores antes
    de lanzar la excepción para que se corrijan de una sola vez.
    """
    # Sin las columnas necesarias no tiene sentido revisar valores.
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        raise BatchError([f"Faltan columnas: {', '.join(missing)}"])

    # Columnas extra (por ejemplo un id) se ignoran; el orden queda fijo.
    clean = frame[RAW_FEATURES + [TARGET]].copy()
    errors = []
    if len(clean) < MIN_BATCH_ROWS:
        errors.append(f"El lote necesita al menos {MIN_BATCH_ROWS} filas; tiene {len(clean)}.")

    # Todas las columnas numéricas (todas menos ocean_proximity) más el objetivo.
    for column in RAW_FEATURES[:-1] + [TARGET]:
        # errors="coerce" convierte el texto inválido en NaN; comparándolo con
        # las celdas que no estaban vacías se distingue "abc" de una celda vacía.
        values = pd.to_numeric(clean[column], errors="coerce")
        not_numeric = values.isna() & clean[column].notna()
        if not_numeric.any():
            errors.append(f"{column}: valores no numéricos en las líneas {_csv_lines(not_numeric)}")
        # Solo total_bedrooms puede faltar; el pipeline lo marca con bedrooms_missing.
        empty = clean[column].isna()
        if column != "total_bedrooms" and empty.any():
            errors.append(f"{column}: celdas vacías en las líneas {_csv_lines(empty)}")
        clean[column] = values

    # Reglas de dominio: conteos e ingresos no pueden ser negativos y un precio
    # de vivienda debe ser positivo.
    for column in NON_NEGATIVE:
        negative = clean[column] < 0
        if negative.any():
            errors.append(f"{column}: valores negativos en las líneas {_csv_lines(negative)}")
    not_positive = clean[TARGET] <= 0
    if not_positive.any():
        errors.append(f"{TARGET}: debe ser mayor que 0 (líneas {_csv_lines(not_positive)})")

    # La categoría debe ser una de las que conoce prepare_pycaret_features.
    unknown = ~clean["ocean_proximity"].isin(list(OCEAN_CODES))
    if unknown.any():
        values = sorted(clean.loc[unknown, "ocean_proximity"].astype(str).unique())[:5]
        errors.append(
            f"ocean_proximity: categorías desconocidas {values}; "
            f"se aceptan {sorted(OCEAN_CODES)}"
        )

    if errors:
        raise BatchError(errors)
    return clean


class ModelStore:
    """Sirve el campeón y lo reemplaza solo si un retador lo supera en el mismo holdout.

    Concurrencia: uvicorn atiende las rutas síncronas en varios hilos y el
    reentrenamiento corre en otro hilo. Por eso:
    - `model` y `version` se reemplazan con una sola asignación (atómica en
      Python); quien ya tomó una referencia al modelo viejo termina con él.
    - `_state_lock` serializa las escrituras de state.json.
    - `_log_lock` evita que dos predicciones mezclen líneas en predictions.jsonl.
    - `_retrain_lock` garantiza un solo entrenamiento a la vez.
    """

    def __init__(
        self,
        data_dir: str | Path,
        base_data_path: str | Path,
        seed_model_path: str | Path | None = None,
        pipeline_factory=default_pipeline,
    ):
        """No toca el disco: los directorios se crean al primer uso.

        Args:
            data_dir: directorio del estado (el volumen en la nube).
            base_data_path: CSV original; aporta la partición base de
                entrenamiento y evaluación.
            seed_model_path: .pkl exportado por el cuaderno, si existe.
            pipeline_factory: crea el pipeline inicial; las pruebas pasan uno
                más pequeño para ir rápido.
        """
        self.data_dir = Path(data_dir)
        self.base_data_path = Path(base_data_path)
        self.seed_model_path = Path(seed_model_path) if seed_model_path else None
        self.pipeline_factory = pipeline_factory

        # La API lee estos atributos en cada solicitud; se reemplazan de una vez.
        self.model = None
        self.version = None

        self._state_lock = threading.Lock()
        self._log_lock = threading.Lock()
        # Un solo entrenamiento a la vez: el arranque inicial también lo toma.
        self._retrain_lock = threading.Lock()
        # Descripción del entrenamiento en curso (None si no hay ninguno). Vive
        # en memoria a propósito: si el proceso se reinicia a mitad de un
        # entrenamiento, no queda un "running" huérfano en state.json.
        self._running = None
        self._thread = None

    @classmethod
    def from_env(cls) -> "ModelStore":
        """Configuración por variables de entorno (ver Dockerfile y render.yaml).

        En local, sin variables, el estado se guarda en ./runtime (en .gitignore).
        """
        return cls(
            data_dir=os.getenv("DATA_DIR", "runtime"),
            base_data_path=os.getenv("BASE_DATA_PATH", "1553768847-housing.csv"),
            seed_model_path=os.getenv("SEED_MODEL_PATH", "models/pycaret_housing.pkl"),
        )

    # Rutas dentro de DATA_DIR (ver el diagrama del docstring del módulo).

    @property
    def champion_path(self) -> Path:
        return self.data_dir / "models" / "champion.pkl"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def predictions_path(self) -> Path:
        return self.data_dir / "predictions.jsonl"

    # ── Estado persistente ──────────────────────────────────────────────

    def read_state(self) -> dict:
        """Lee state.json; si aún no existe devuelve el estado vacío.

        Claves:
            champion            datos del modelo que sirve (versión, métricas…)
            adopted_seed_mtime  fecha del último .pkl del cuaderno adoptado
            batches             lotes subidos y si ya se usaron para entrenar
            history             todos los entrenamientos y solicitudes de datos
            last_retrain        el más reciente de history (lo usa /retrain/status)
            data_requested_at   cuándo un reentrenamiento se omitió por falta de
                                datos; vuelve a None al subir un lote
        """
        if not self.state_path.exists():
            return {
                "champion": None,
                "adopted_seed_mtime": None,
                "batches": [],
                "history": [],
                "last_retrain": None,
                "data_requested_at": None,
            }
        return json.loads(self.state_path.read_text())

    def _update_state(self, change) -> dict:
        """Aplica `change(state)` y reescribe state.json de forma atómica.

        Se escribe primero un archivo temporal y luego se renombra: si el
        proceso muere a mitad de la escritura, state.json queda intacto (el
        renombrado es atómico en el mismo disco).
        """
        with self._state_lock:
            state = self.read_state()
            change(state)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False))
            os.replace(temporary, self.state_path)
            return state

    # ── Datos ───────────────────────────────────────────────────────────

    def _split(self, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Parte un CSV en 80 % entrenamiento / 20 % evaluación, siempre igual.

        Con la misma semilla, el mismo archivo da exactamente las mismas filas en
        cada parte, en cada reentrenamiento y en cada reinicio.
        """
        return train_test_split(pd.read_csv(path), test_size=EVAL_SIZE, random_state=SEED)

    def _datasets(self, batch_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Entrenamiento y evaluación: partición base + la de cada lote subido.

        Cada lote se parte siempre igual, así que sus filas de evaluación nunca
        llegan a entrenar a un modelo y campeón y retador se miden en filas que
        ninguno vio. Evaluar también con filas de los lotes hace que la
        comparación refleje los datos recientes, no solo los de 1990.
        """
        splits = [self._split(self.base_data_path)]
        splits += [self._split(self.uploads_dir / f"{batch_id}.csv") for batch_id in batch_ids]
        train = pd.concat([split[0] for split in splits], ignore_index=True)
        evaluation = pd.concat([split[1] for split in splits], ignore_index=True)
        return train, evaluation

    def add_batch(self, frame: pd.DataFrame) -> dict:
        """Valida y guarda un lote etiquetado; queda pendiente hasta el próximo reentrenamiento."""
        clean = validate_batch(frame)
        # El id ordena los lotes por fecha; el sufijo aleatorio evita choques si
        # se suben dos lotes en el mismo segundo.
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        clean.to_csv(self.uploads_dir / f"{batch_id}.csv", index=False)

        batch = {
            "id": batch_id,
            "uploaded_at": utc_now(),
            "rows": len(clean),
            "used_in_training": False,
        }

        def change(state):
            state["batches"].append(batch)
            # Ya hay datos: se quita el aviso "Se necesitan datos nuevos".
            state["data_requested_at"] = None

        self._update_state(change)
        logger.info("Lote %s guardado con %d filas", batch_id, len(clean))
        return batch

    def pending_batches(self) -> list[dict]:
        """Lotes que ningún reentrenamiento ha usado todavía."""
        return [b for b in self.read_state()["batches"] if not b["used_in_training"]]

    # ── Modelo ──────────────────────────────────────────────────────────

    def _install(self, model, record: dict) -> dict:
        """Guarda el campeón (escritura atómica) y lo pone a servir.

        Primero se escribe en disco y después se cambia `self.model`: si el
        guardado falla, la API sigue sirviendo el campeón anterior.
        """
        self.champion_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.champion_path.with_suffix(".tmp")
        joblib.dump(model, temporary)
        os.replace(temporary, self.champion_path)
        self.model = model
        self.version = record["version"]
        return record

    def _champion_record(self, state: dict, model, trigger: str, metrics: dict, train_rows: int) -> dict:
        """Ficha del nuevo campeón: versión consecutiva (v1, v2…), origen y métricas."""
        number = (state["champion"] or {}).get("number", 0) + 1
        return {
            "number": number,
            "version": f"v{number}",
            "trained_at": utc_now(),
            # "inicial", "cuaderno", "cron" o "manual".
            "trigger": trigger,
            # Nombre del último paso del pipeline, p. ej. RandomForestRegressor.
            "algorithm": type(model.steps[-1][1]).__name__ if hasattr(model, "steps") else type(model).__name__,
            "train_rows": train_rows,
            "metrics": metrics,
        }

    def load_or_bootstrap(self):
        """Carga el campeón; si no hay, adopta el .pkl del cuaderno o entrena uno.

        La API lo ejecuta en un hilo al arrancar. Toma `_retrain_lock` para que
        nadie lance un reentrenamiento mientras se prepara el modelo. Si algo
        falla, se registra el error y la API sigue arriba respondiendo 503 en
        /health, en lugar de caerse en un ciclo de reinicios.
        """
        with self._retrain_lock:
            try:
                self._load_or_bootstrap()
            except Exception:
                logger.exception("No se pudo preparar el modelo")

    def _load_or_bootstrap(self):
        state = self.read_state()
        # Lotes que ya forman parte del campeón: se usan si hay que entrenar de
        # nuevo desde cero (por ejemplo, si se borró champion.pkl).
        used = [b["id"] for b in state["batches"] if b["used_in_training"]]
        seed = self.seed_model_path
        # La fecha de modificación identifica cada exportación del cuaderno.
        seed_mtime = seed.stat().st_mtime if seed and seed.exists() else None

        # Caso 1: un .pkl recién exportado por el cuaderno reemplaza al campeón
        # una sola vez; después solo vuelve a adoptarse si se exporta de nuevo.
        if seed_mtime and seed_mtime != state.get("adopted_seed_mtime"):
            model, trigger, train_rows = joblib.load(seed), "cuaderno", None
            logger.info("Adoptando el modelo exportado por el cuaderno: %s", seed)
        # Caso 2 (el habitual tras un reinicio): ya hay campeón en el volumen.
        elif self.champion_path.exists() and state["champion"]:
            self.model = joblib.load(self.champion_path)
            self.version = state["champion"]["version"]
            logger.info("Campeón %s cargado desde %s", self.version, self.champion_path)
            return
        # Caso 3 (primer despliegue en la nube): no hay ningún modelo.
        else:
            logger.info("No hay modelo: entrenando el pipeline inicial con los datos base")
            train, _ = self._datasets(used)
            model = self.pipeline_factory().fit(prepare_pycaret_features(train), train[TARGET])
            trigger, train_rows = "inicial", len(train)

        # Casos 1 y 3: medir el nuevo campeón, guardarlo y anotarlo en el historial.
        _, evaluation = self._datasets(used)
        metrics = evaluate(model, evaluation)
        record = self._champion_record(state, model, trigger, metrics, train_rows)
        self._install(model, record)

        def change(state):
            state["champion"] = record
            state["history"].append({**record, "status": "promoted", "finished_at": record["trained_at"]})
            if trigger == "cuaderno":
                state["adopted_seed_mtime"] = seed_mtime

        self._update_state(change)
        logger.info("Campeón %s listo (%s) · RMSE %s", record["version"], trigger, metrics["rmse"])

    # ── Reentrenamiento ─────────────────────────────────────────────────

    def start_retrain(self, trigger: str) -> dict:
        """Lanza el reentrenamiento en un hilo y responde sin esperar.

        Devuelve un dict cuyo `status` la API traduce a código HTTP:
            started        202: hay lotes nuevos y comenzó el entrenamiento
            awaiting_data  200: no hay lotes nuevos; se registra la solicitud
            running        409: ya hay un entrenamiento en curso
            not_ready      503: el modelo inicial aún no está listo
        """
        if self.model is None:
            return {"status": "not_ready", "message": "El modelo inicial todavía no está listo."}
        # acquire(blocking=False) devuelve False de inmediato si otro hilo ya
        # entrena, en vez de dejar la petición HTTP esperando.
        if not self._retrain_lock.acquire(blocking=False):
            return {"status": "running", "message": "Ya hay un entrenamiento en curso."}
        if not self.pending_batches():
            self._retrain_lock.release()
            return self._record_awaiting_data(trigger)

        # El hilo hereda el candado y lo libera al terminar, pase lo que pase.
        def run():
            try:
                self._retrain(trigger)
            finally:
                self._retrain_lock.release()

        self._running = {"status": "running", "trigger": trigger, "started_at": utc_now()}
        # daemon=True: el hilo no impide que el proceso se apague.
        self._thread = threading.Thread(target=run, name="retrain", daemon=True)
        self._thread.start()
        return {**self._running, "status": "started"}

    def retrain(self, trigger: str) -> dict:
        """Versión síncrona de start_retrain (pruebas y uso desde consola)."""
        with self._retrain_lock:
            if not self.pending_batches():
                return self._record_awaiting_data(trigger)
            self._running = {"status": "running", "trigger": trigger, "started_at": utc_now()}
            return self._retrain(trigger)

    def retrain_status(self) -> dict:
        """El entrenamiento en curso o, si no hay, el resultado del último."""
        if self._running:
            return self._running
        return self.read_state()["last_retrain"] or {"status": "idle"}

    def _record_awaiting_data(self, trigger: str) -> dict:
        """Sin datos nuevos no se reentrena: se registra la solicitud de datos.

        Reentrenar con los mismos datos produciría el mismo modelo, así que en
        lugar de gastar cómputo se deja constancia (historial + aviso en la
        interfaz + advertencia en los logs del cron) de que hacen falta datos.
        """
        record = {
            "status": "awaiting_data",
            "trigger": trigger,
            "finished_at": utc_now(),
            "message": "No hay datos nuevos: se omitió el reentrenamiento. Suba un CSV etiquetado.",
        }

        def change(state):
            state["history"].append(record)
            state["last_retrain"] = record
            state["data_requested_at"] = record["finished_at"]

        self._update_state(change)
        logger.warning("⚠️ SE NECESITAN DATOS NUEVOS: reentrenamiento (%s) omitido", trigger)
        return record

    def _retrain(self, trigger: str) -> dict:
        """Entrena el retador, lo compara con el campeón y decide. Requiere `_retrain_lock`."""
        state = self.read_state()
        # El retador aprende con TODOS los lotes (los ya usados y los nuevos);
        # `new_ids` solo sirve para marcar después cuáles se consumieron.
        batch_ids = [b["id"] for b in state["batches"]]
        new_ids = [b["id"] for b in state["batches"] if not b["used_in_training"]]
        champion = self.model
        logger.info("Reentrenando (%s) con %d lotes nuevos", trigger, len(new_ids))

        try:
            train, evaluation = self._datasets(batch_ids)
            # clone conserva el algoritmo e hiperparámetros del campeón sin sus pesos.
            challenger = clone(champion).fit(prepare_pycaret_features(train), train[TARGET])
            # Ambos se miden ahora y en las mismas filas: comparación justa.
            champion_metrics = evaluate(champion, evaluation)
            challenger_metrics = evaluate(challenger, evaluation)
        except Exception as error:
            # Un error (p. ej. falta de memoria) no tumba la API: el campeón
            # sigue sirviendo y el fallo queda en el historial.
            logger.exception("Falló el reentrenamiento")
            record = {"status": "failed", "trigger": trigger, "finished_at": utc_now(), "error": str(error)}
        else:
            # Empate = se queda el campeón: cambiar de modelo tiene un costo
            # (riesgo) y no aporta beneficio.
            promoted = challenger_metrics["rmse"] < champion_metrics["rmse"]
            record = {
                "status": "promoted" if promoted else "kept",
                "trigger": trigger,
                "finished_at": utc_now(),
                "new_batches": new_ids,
                "train_rows": len(train),
                "eval_rows": len(evaluation),
                "champion_metrics": champion_metrics,
                "challenger_metrics": challenger_metrics,
            }
            if promoted:
                champion_record = self._champion_record(
                    state, challenger, trigger, challenger_metrics, len(train)
                )
                self._install(challenger, champion_record)
                record["version"] = champion_record["version"]

        def change(state):
            if record["status"] != "failed":
                # Los lotes se consumen aunque gane el campeón: repetir con los
                # mismos datos daría el mismo resultado. Si falló, siguen
                # pendientes para el próximo intento.
                for batch in state["batches"]:
                    if batch["id"] in new_ids:
                        batch["used_in_training"] = True
            if record["status"] == "promoted":
                state["champion"] = champion_record
            state["history"].append(record)
            state["last_retrain"] = record

        self._update_state(change)
        self._running = None
        self._log_to_mlflow(record)
        logger.info(
            "Reentrenamiento %s · campeón RMSE %s · retador RMSE %s",
            record["status"],
            record.get("champion_metrics", {}).get("rmse"),
            record.get("challenger_metrics", {}).get("rmse"),
        )
        return record

    def _log_to_mlflow(self, record: dict):
        """Registra la decisión en MLflow si hay servidor configurado (opcional en la nube).

        Solo se guardan parámetros y métricas, no el modelo: subir 138 MB en cada
        reentrenamiento sería lento y el modelo ya está en el volumen. Un error
        de MLflow nunca debe deshacer un reentrenamiento exitoso, por eso se
        captura cualquier excepción.
        """
        if not os.getenv("MLFLOW_TRACKING_URI") or record["status"] == "failed":
            return
        try:
            import mlflow

            mlflow.set_experiment("California_Housing_Production_Retrain")
            with mlflow.start_run(run_name=f"retrain_{record['finished_at']}"):
                mlflow.log_params(
                    {
                        "trigger": record["trigger"],
                        "decision": record["status"],
                        "train_rows": record["train_rows"],
                        "eval_rows": record["eval_rows"],
                        "new_batches": len(record["new_batches"]),
                    }
                )
                # champion_rmse, challenger_rmse, champion_mae, …
                for role in ("champion", "challenger"):
                    mlflow.log_metrics({f"{role}_{k}": v for k, v in record[f"{role}_metrics"].items()})
        except Exception:
            logger.exception("No se pudo registrar el reentrenamiento en MLflow")

    # ── Monitoreo ───────────────────────────────────────────────────────

    @cached_property
    def reference_stats(self) -> dict:
        """Media y desviación de la partición de entrenamiento base.

        Se calcula una sola vez (cached_property) en la primera predicción.
        """
        train, _ = self._split(self.base_data_path)
        numeric = train[RAW_FEATURES].select_dtypes("number")
        return {column: (numeric[column].mean(), numeric[column].std()) for column in numeric}

    def outliers(self, features: dict) -> list[dict]:
        """Valores a más de 3 desviaciones de la media (misma regla que monitor.py).

        Mira una sola vivienda: sirve para avisar que esa predicción es poco
        confiable, no para afirmar que cambió la distribución de los datos. El
        porcentaje de atípicas en /monitor/summary da esa señal agregada.
        """
        found = []
        for feature, (mean, std) in self.reference_stats.items():
            value = features.get(feature)
            # total_bedrooms puede venir vacío (None) y std=0 dividiría por cero.
            if value is not None and std > 0:
                z_score = abs(value - mean) / std
                if z_score > Z_SCORE_THRESHOLD:
                    found.append({"feature": feature, "z_score": round(float(z_score), 2)})
        return found

    def log_prediction(self, features: dict, prediction: float) -> list[dict]:
        """Agrega la predicción a predictions.jsonl, que sobrevive a reinicios.

        Formato JSON Lines: un objeto JSON por línea. Agregar una línea es barato
        y el archivo se puede leer con pandas.read_json(lines=True).
        """
        outliers = self.outliers(features)
        record = {
            "timestamp": utc_now(),
            # Permite comparar el comportamiento de cada versión del modelo.
            "model_version": self.version,
            "input": features,
            "prediction": round(prediction, 2),
            "outliers": [o["feature"] for o in outliers],
        }
        with self._log_lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.predictions_path, "a") as log_file:
                log_file.write(json.dumps(record) + "\n")
        if outliers:
            logger.warning("⚠️ Entrada atípica en: %s", [o["feature"] for o in outliers])
        return outliers

    def prediction_summary(self) -> dict:
        """Resumen de todas las predicciones registradas y de las últimas 24 h.

        Lee el archivo completo en cada llamada: es suficiente para el volumen
        de un curso (cien mil predicciones ≈ 30 MB). Con más tráfico convendría
        una base de datos o rotar el archivo.
        """
        if not self.predictions_path.exists():
            return {"total_predictions": 0}
        # convert_dates=False: las fechas se convierten abajo de forma explícita.
        records = pd.read_json(self.predictions_path, lines=True, convert_dates=False)
        if records.empty:
            return {"total_predictions": 0}
        timestamps = pd.to_datetime(records["timestamp"], utc=True)
        last_day = records[timestamps >= datetime.now(timezone.utc) - timedelta(days=1)]
        return {
            "total_predictions": len(records),
            "last_24h": len(last_day),
            "avg_prediction": round(float(records["prediction"].mean()), 2),
            "min_prediction": round(float(records["prediction"].min()), 2),
            "max_prediction": round(float(records["prediction"].max()), 2),
            # Fracción de predicciones con al menos una variable atípica. Si
            # sube con el tiempo, los datos de entrada se están alejando de los
            # de entrenamiento y conviene conseguir datos nuevos.
            "outlier_rate": round(float(records["outliers"].map(bool).mean()), 4),
            "outlier_rate_24h": round(float(last_day["outliers"].map(bool).mean()), 4) if len(last_day) else 0.0,
            "last_prediction_at": records["timestamp"].iloc[-1],
        }

    def status(self) -> dict:
        """Todo lo que muestra la interfaz en una sola respuesta (/status).

        Los lotes y el historial se devuelven del más reciente al más antiguo y
        limitados a 20 para que la respuesta no crezca sin límite.
        """
        state = self.read_state()
        pending = [b for b in state["batches"] if not b["used_in_training"]]
        return {
            "model_ready": self.model is not None,
            "champion": state["champion"],
            "retrain": self.retrain_status(),
            "pending_batches": len(pending),
            "pending_rows": sum(b["rows"] for b in pending),
            "data_requested_at": state["data_requested_at"],
            "batches": state["batches"][-20:][::-1],
            "history": state["history"][-20:][::-1],
        }
