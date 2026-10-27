# 🏠 MLOps Pipeline — California Housing Prediction

<div align="center">

![Miku MLOps](miku_mlops.png)

*✨ Pipeline completo de Machine Learning Operations ✨*

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/badge/uv-Dependency%20Manager-DE5FE9?logo=uv)](https://docs.astral.sh/uv/)
[![MLFlow](https://img.shields.io/badge/MLFlow-Experiment%20Tracking-0194E2?logo=mlflow)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?logo=scikit-learn)](https://scikit-learn.org)

</div>

---

## 📋 Descripción

Pipeline end-to-end de MLOps para predecir el **valor medio de viviendas en California** usando el dataset [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (20,640 registros).

El proyecto cubre todo el ciclo de vida de un modelo de ML:

```
uv → EDA → Pipeline → Cross-Validation → MLFlow → Parquet → Docker → FastAPI
```

---

## 🗂️ Estructura del Proyecto

```
mlops/
├── MANUAL.md                   # 📘 Manual paso a paso: desplegar y operar la API
├── train.py                    # 🤖 10 modelos; selección por CV 5-Fold
├── app.py                      # 🚀 API REST con FastAPI
├── pycaret_app.py              # 🚀 API PyCaret: interfaz, monitoreo, subida de datos y reentrenamiento
├── model_store.py              # 🏆 Modelo campeón, lotes subidos e historial en DATA_DIR
├── cron_job.py                 # ⏰ Cron jobs: revisión de salud y reentrenamiento quincenal
├── .env.example                # 🔑 Plantilla de .env (ADMIN_TOKEN para Docker Compose)
├── static/index.html           # 🖥️ Interfaz web servida en /
├── pycaret_features.py         # 🧩 Esquema compartido notebook/API
├── monitor.py                  # 📈 Valores atípicos y registro de predicciones
├── feature_store.py            # 🧮 Feature Store centralizado
├── retrain.py                  # 🔄 Reentrenamiento automático
├── eda.ipynb                   # 📊 EDA tradicional + Sweetviz
├── pycaret_experimentos.ipynb  # 🤖 PyCaret + corridas MLflow + exportación
├── reportes/                   # 📑 Reportes HTML de Sweetviz (generados por eda.ipynb)
├── 1553768847-housing.csv      # 📁 Dataset original
├── pyproject.toml              # 📦 Dependencias (uv)
├── uv.lock                     # 🔒 Versiones exactas
├── .python-version             # 🐍 Python 3.12
├── Dockerfile                  # 🐳 Imagen multi-stage
├── compose.yml                 # 🐳 Docker Compose (API + cron + MLFlow + Jupyter)
├── render.yaml                 # ☁️ Blueprint de Render (API + 2 cron jobs)
├── railway.json                # ☁️ Configuración de Railway para la API
├── deploy/railway.*.json       # ☁️ Configuración de Railway para cada cron job
├── model_metrics.json          # 📋 Métricas del modelo actual
├── tests/
│   ├── test_pipeline.py        # 🧪 Pipeline original
│   ├── test_pycaret_serving.py # 🧪 Esquema y API PyCaret
│   └── test_retraining.py      # 🧪 Subida de datos, reentrenamiento y cron jobs
├── .github/workflows/
│   ├── ci_cd.yml               # ⚙️ CI/CD: lint → train → tests → GitHub Pages
│   └── retrain.yml             # 🔄 Reentrenamiento semanal como artefacto
├── mlops.html                  # 🎓 Presentación 1: Intro a MLOps
├── index.html                  # 🎓 Presentación 2: Pipeline práctico
└── presentacion_mlops_avanzado.html  # 🎓 Presentación 3: MLOps avanzado
```

---

## 🚀 Inicio Rápido

### 1. Instalar dependencias

```bash
# Instalar uv (si no lo tienes)
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Crear el entorno con las versiones exactas de uv.lock
# (uv descarga Python 3.12 automáticamente si no lo tienes)
uv sync
```

`uv sync` instala tres grupos de dependencias:

| Grupo | Contenido | Dónde se usa |
|-------|-----------|--------------|
| *(principal)* | pandas, scikit-learn, mlflow, fastapi, uvicorn, pyarrow | Siempre (también en Docker) |
| `dev` | pytest, flake8, httpx2 | Tests y CI |
| `eda` | jupyter, matplotlib, seaborn, sweetviz, nbformat | Notebook `eda.ipynb` |
| `pycaret` *(opcional)* | PyCaret 4.0.0a8 | Notebook `pycaret_experimentos.ipynb` |

Para instalar solo lo necesario para producción: `uv sync --no-default-groups`.

### 2. Entrenar los modelos

```bash
uv run python train.py
```

Esto entrena **10 modelos de regresión** con **validación cruzada 5-Fold** y registra todo en MLFlow. El mejor se elige por el RMSE promedio de CV; el conjunto de prueba se reserva para informar el resultado final:

| Modelo | RMSE Test ($) | CV RMSE μ ($) | CV σ ($) | R² |
|--------|--------------|---------------|----------|------|
| 🏆 **Gradient Boosting** | **48,107** | **48,323** | **1,006** | **0.8234** |
| Random Forest | 48,942 | 49,262 | 624 | 0.8172 |
| Extra Trees | 52,281 | 52,374 | 667 | 0.7914 |
| KNN | 61,310 | 61,732 | 742 | 0.7132 |
| Decision Tree | 66,557 | 67,226 | 1,254 | 0.6620 |
| Linear Regression | 70,059 | 68,623 | 1,439 | 0.6254 |
| Ridge | 70,066 | 68,622 | 1,440 | 0.6254 |
| Lasso | 70,060 | 68,623 | 1,439 | 0.6254 |
| ElasticNet | 78,389 | 78,285 | 1,030 | 0.5311 |
| AdaBoost | 90,406 | 89,439 | 3,264 | 0.3763 |

### 3. Ver experimentos en MLFlow

```bash
uv run mlflow ui --port 5001
# Abre http://localhost:5001
```

> ⚠️ En macOS no uses el puerto 5000 (el predeterminado de MLflow): lo ocupa el Receptor AirPlay, y en `localhost:5000` responde AirPlay en vez de MLflow, así que no se ven los experimentos.

### 4. Levantar la API

```bash
uv run uvicorn app:app --reload
# Abre http://localhost:8000/docs para Swagger UI
```

### 5. Hacer una predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'
```

---

## 🔁 De Poetry a uv

| Poetry | uv |
|--------|----|
| `poetry install` | `uv sync` |
| `poetry add pkg` | `uv add pkg` |
| `poetry add --group dev pkg` | `uv add --dev pkg` |
| `poetry run cmd` | `uv run cmd` |
| `poetry.lock` | `uv.lock` |

---

## 🔬 Análisis Exploratorio (EDA)

El notebook `eda.ipynb` contiene **15 secciones**: 13 de EDA tradicional y 2 de EDA automatizado con Sweetviz. Los experimentos de PyCaret están en su propio cuaderno, `pycaret_experimentos.ipynb`:

1. **Información general** del dataset
2. **Estadísticas descriptivas** con heatmap
3. **Valores nulos** — `total_bedrooms` tiene 207 (~1%)
4. **Variable objetivo** — histograma, boxplot, QQ-plot
5. **Distribución** de todas las variables numéricas
6. **Asimetría (skewness)** — detectar variables sesgadas
7. **Matriz de correlación** triangular
8. **Ingreso vs Valor** — scatter + mapa de densidad
9. **Proximidad al océano** — pie, boxplot, violin
10. **Distribución geográfica** — mapas de California
11. **Detección de outliers** con IQR
12. **Feature Engineering** — variables derivadas
13. **Conclusiones** del análisis
14. **Sweetviz** — reporte general, Train vs Test (misma partición de `train.py`) e Interior vs Costa
15. **Tradicional vs Sweetviz** — verificación de que ambos coinciden y tabla comparativa

Los reportes HTML de Sweetviz quedan en `reportes/` y se enlazan desde la presentación.

```bash
uv run jupyter notebook eda.ipynb
```

### Experimentos PyCaret

PyCaret `4.0.0a8` está fijado como grupo opcional en `uv.lock`. Es una versión preliminar compatible con Python 3.12 y el scikit-learn del proyecto; usa `RegressionExperiment`. El notebook registra cada candidato explícitamente en MLflow, porque el registro automático de PyCaret 3 no forma parte de la API 4.

```bash
uv sync --locked --group pycaret
uv run --group pycaret jupyter notebook pycaret_experimentos.ipynb
```

El cuaderno está organizado en pasos, cada uno explicado antes de su código:

1. **Partir los datos** 80/20 (misma semilla que `train.py`) y guardarlos en `data/train.csv` y `data/test.csv`; es lo primero del cuaderno
2. **Librerías y configuración**
3. **Preparar las columnas** con `prepare_pycaret_features`, la misma función de [pycaret_app.py](pycaret_app.py); el cuaderno muestra su código
4. **Probar 21 modelos** de PyCaret (lineales, robustos, KNN, SVM, árboles, ensambles, MLP y un `dummy` de referencia) con columnas normalizadas y validación cruzada de 5 partes
5. **Guardar una corrida de MLflow por modelo** (hiperparámetros + métricas de CV), consultarlas y **abrir la interfaz de MLflow** en http://localhost:5001
6. **Evaluar el mejor** con `data/test.csv`, que no se usó para elegirlo
7. **Exportar** `models/pycaret_housing.pkl` y guardar la corrida `selected_final_model` con las métricas de prueba y el modelo

Para servir el modelo localmente:

```bash
uv run uvicorn pycaret_app:app --reload
```

La API original de `app.py` y su archivo `best_model.pkl` siguen disponibles para el flujo manual anterior.

---

## 🔄 Validación Cruzada (Cross-Validation)

Cada modelo se evalúa con **5-Fold Cross-Validation** sobre el set de entrenamiento:

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
cv_rmse = np.sqrt(-cv_scores)
```

Las métricas de CV se registran en MLFlow:
- `cv_rmse_mean` — RMSE promedio de los 5 folds
- `cv_rmse_std` — Desviación estándar (estabilidad)
- `cv_rmse_fold_1..5` — RMSE individual por fold

---

## ⚙️ Pipeline de Preprocesamiento

Se usa `ColumnTransformer` + `Pipeline` para **evitar data leakage**:

```python
# Numérico: impute con mediana → escalar
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categórico: impute con moda → one-hot encoding
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar ambos
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])
```

> ⚠️ **Data Leakage**: Si escalas antes de dividir, el `StandardScaler` aprende estadísticas del test set, generando métricas falsamente optimistas.

---

## 📈 Monitoreo en Producción

El módulo `monitor.py` implementa:

- **Registro en memoria** de cada predicción con timestamp; vuelca lotes de 100 a un archivo local
- **Detección de valores individuales atípicos** usando z-score (umbral: 3σ). La ruta se llama `check-drift` por compatibilidad, pero no compara distribuciones.
- **Resumen estadístico** de las predicciones del proceso actual

```bash
# Comprobar valores atípicos
curl -X POST http://localhost:8000/monitor/check-drift \
  -H "Content-Type: application/json" \
  -d '{"longitude":-122.23, "latitude":37.88, ...}'

# Ver resumen
curl http://localhost:8000/monitor/summary
```

Este monitor pertenece a la API original (`app.py`): guarda el registro en memoria y no dispara el reentrenamiento ni mide error de predicción, porque la API no recibe el valor real de la vivienda. La API PyCaret desplegada resuelve esas limitaciones con un ciclo propio (ver [Producción: monitoreo y reentrenamiento quincenal](#️-producción-monitoreo-y-reentrenamiento-quincenal)). El workflow semanal ejecuta `retrain.py` y publica `best_model.pkl` junto con sus métricas como artefacto de GitHub Actions; no despliega el modelo en la API. Para usarlo localmente, descarga el artefacto y reinicia la API. La comparación de RMSE en `retrain.py` usa el holdout del CSV y solo es comparable si los datos y la partición de evaluación siguen siendo los mismos.

---

## ♻️ Producción: monitoreo y reentrenamiento quincenal

> 📘 **¿Cómo desplegarla y operarla paso a paso?** Ver el [MANUAL.md](MANUAL.md): pruebas locales, Railway, Render, formato del CSV, lectura de logs y solución de problemas.

La API PyCaret (`pycaret_app.py`) sirve una interfaz web, registra cada predicción en disco y se reentrena con datos etiquetados que suben los usuarios. Dos cron jobs la llaman por HTTP:

```
                 ┌──────────── Servicio API (FastAPI + volumen /data) ────────────┐
 Usuario ──────► │ /               interfaz: predecir, subir CSV, estado, historial│
                 │ /predict        predicción + registro en predictions.jsonl      │
                 │ /data/upload    CSV etiquetado (token)                          │
                 │ /retrain        retador en segundo plano (token)                │
                 └────────────────────────────▲──────────────────────────────────┘
   cron health   cada 15 min ─────────────────┤ /health + predicción de prueba
   cron retrain  días 1 y 15, 06:00 UTC ──────┘ ¿hay lotes nuevos? sí → reentrena · no → pide datos
```

**Ciclo de reentrenamiento** (`model_store.py`):

1. Quien tenga el valor real de viviendas nuevas sube un CSV con las 9 columnas de entrada y `median_house_value`. Si falta una columna, hay texto en una columna numérica, valores negativos o una categoría desconocida, se rechaza el lote completo y se indica cada línea con error.
2. El cron `retrain` llama a `POST /retrain?trigger=cron`. **Si nadie subió datos, no se entrena**: queda registrada la solicitud y la interfaz muestra "Se necesitan datos nuevos" hasta que alguien suba un lote.
3. Con datos nuevos, `clone(campeón)` entrena un **retador** con la partición base de entrenamiento más el 80 % de cada lote. Se conservan el algoritmo y los hiperparámetros elegidos en el cuaderno; no hace falta PyCaret en producción.
4. Campeón y retador se evalúan en el **mismo holdout**: el `data/test.csv` del cuaderno (misma semilla) más el 20 % de cada lote. Ninguno de los dos entrenó con esas filas.
5. El retador reemplaza al campeón solo si tiene menor RMSE. El cambio es inmediato, sin reiniciar la API, y queda en el historial (y en MLflow si `MLFLOW_TRACKING_URI` está definido).

**Al arrancar**, la API carga el campeón del volumen. Si no hay campeón, adopta el `.pkl` exportado por el cuaderno (`SEED_MODEL_PATH`) o, en la nube, donde `models/` no está en Git, entrena el pipeline inicial; mientras tanto `/health` responde 503. Un `.pkl` que el cuaderno exporte más tarde reemplaza al campeón en el siguiente reinicio.

**Monitoreo:** `/monitor/summary` lee `predictions.jsonl`, que sobrevive a reinicios. Muestra total, últimas 24 h, promedio y el porcentaje de entradas atípicas (z-score > 3 frente al entrenamiento). Las predicciones de prueba del cron llevan `X-Health-Check: true` y no cuentan.

| Ruta | Token | Uso |
|------|:-----:|-----|
| `GET /` | | Interfaz web |
| `GET /health` | | 200 con la versión del modelo, 503 sin modelo |
| `POST /predict` | | Predicción + entradas atípicas |
| `GET /status` | | Campeón, lotes pendientes, historial |
| `GET /monitor/summary` | | Resumen de predicciones |
| `POST /data/upload` | ✅ | Cuerpo `text/csv` |
| `POST /retrain` | ✅ | 202 iniciado · 200 sin datos · 409 en curso |
| `GET /retrain/status` | | En curso o último resultado |

El token va en el encabezado `X-Admin-Token` y debe ser igual a la variable `ADMIN_TOKEN`; sin ella, las rutas de administración responden 503.

### Probar en local

```bash
ADMIN_TOKEN=secreto uv run uvicorn pycaret_app:app --reload   # estado en ./runtime
# http://localhost:8000

export API_URL=http://localhost:8000 ADMIN_TOKEN=secreto
uv run python cron_job.py health
curl -X POST localhost:8000/data/upload -H "X-Admin-Token: secreto" \
     -H "Content-Type: text/csv" --data-binary @nuevos.csv
uv run python cron_job.py retrain
```

### Desplegar en Railway

1. **API:** *New Project → Deploy from GitHub repo*. Railway usa `Dockerfile` y `railway.json` (health check en `/health`).
   - *Variables:* `ADMIN_TOKEN` (un valor largo y aleatorio).
   - *Volume:* montarlo en `/data`.
   - *Networking:* *Generate Domain*.
2. **Cron de salud:** *New → GitHub Repo* (el mismo). En *Settings → Config-as-code* usa `deploy/railway.health.json` (o en *Settings* pon *Start Command* `python cron_job.py health` y *Cron Schedule* `*/15 * * * *`). Variable `API_URL=https://${{housing-api.RAILWAY_PUBLIC_DOMAIN}}`, cambiando `housing-api` por el nombre de tu servicio.
3. **Cron de reentrenamiento:** igual, con `deploy/railway.retrain.json` (`python cron_job.py retrain`, `0 6 1,15 * *`). Variables `API_URL` y `ADMIN_TOKEN=${{housing-api.ADMIN_TOKEN}}`.

### Desplegar en Render

*New → Blueprint* y selecciona el repositorio: `render.yaml` crea la API con disco en `/data` y los dos cron jobs. Render pide los valores marcados `sync: false`: el mismo `ADMIN_TOKEN` para la API y el cron de reentrenamiento, y en `API_URL` la URL pública de la API (si aún no la conoces, ponla después en *Environment*).

> ⚠️ **Memoria y costo.** El RandomForest exportado pesa 138 MB y el proceso llega a ~700 MB al reentrenar. En Render se necesita el plan `1c-2g` (el disco persistente y los cron jobs no existen en el plan gratuito); con 512 MB el reentrenamiento se queda sin memoria. En Railway el plan Hobby alcanza. Sin volumen, cada redespliegue borra los lotes subidos y el historial.

> Cron no puede expresar "cada 14 días": `0 6 1,15 * *` se ejecuta los días 1 y 15 de cada mes (06:00 UTC = 01:00 en Colombia).

---

## 🐳 Docker

```bash
# 1. Una sola vez: crear .env con el token de administración (está en .gitignore)
cp .env.example .env
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"   # pégalo en .env

# 2. Construir y levantar todo en segundo plano
#    --build reconstruye las imágenes si cambió el código (con caché, es rápido)
docker compose up -d --build

# Token de Jupyter y mensajes de los cron jobs
docker compose logs notebook
docker compose logs -f cron

# Después de ejecutar pycaret_experimentos.ipynb
docker compose restart api
```

| Servicio | URL | Qué hace |
|----------|-----|----------|
| `api` | http://localhost:8000 | Interfaz, predicciones, subida de datos y reentrenamiento (Swagger en `/docs`) |
| `cron` | — | Revisa la API cada 15 min y pide el reentrenamiento los días 1 y 15 a las 06:00 UTC (`cron_job.py schedule`), igual que los cron jobs de Railway/Render |
| `mlflow` | http://localhost:5050 | Experimentos del cuaderno y registro de cada reentrenamiento |
| `notebook` | http://localhost:8888 | Jupyter con PyCaret |

Compose lee `ADMIN_TOKEN` de `.env` y se detiene con un mensaje si falta. Sin `--build`, Compose reutiliza las imágenes que ya existan aunque el código haya cambiado. Si el puerto 8888 está ocupado (por ejemplo, por un Jupyter fuera de Docker), define `JUPYTER_PORT=8889` en `.env`. Las imágenes se construyen con `uv.lock`: API, cron y MLflow usan dependencias principales; Jupyter añade los grupos `eda` y `pycaret`. El directorio `models/` se comparte entre notebook y API. Si no hay modelo exportado, la API entrena el pipeline inicial al arrancar; después de exportar uno nuevo desde el cuaderno, reinicia la API para que lo adopte. El campeón, los lotes y el historial se guardan en el volumen `api_data`, y MLflow en `mlflow_data`.

Para probar un reentrenamiento sin esperar al día 1 o 15:

```bash
docker compose exec cron python cron_job.py retrain
```

---

## 🧪 Tests

```bash
uv run python train.py        # los tests necesitan best_model.pkl
uv run pytest -v
uv run flake8 *.py tests/ --max-line-length=120
```

38 tests que cubren:
- ✅ Feature Store (features derivadas, preprocesador, info, versiones) — 4
- ✅ Monitor (logging, drift normal, drift extremo, resumen) — 5
- ✅ Integridad de datos (CSV existe, columnas correctas, modelo existe) — 3
- ✅ API (`/health`, `/predict` con el modelo de `train.py` y el de `retrain.py`, modelo ausente → 503, payload inválido → 422, valores atípicos) — 6
- ✅ Serialización (los modelos se pueden guardar en MLflow con skops) — 2
- ✅ PyCaret (esquema de features, inferencia y errores de la API) — 3
- ✅ Reentrenamiento (arranque, validación de lotes, solicitud de datos, campeón vs. retador, token, interfaz, cron jobs y programador de Compose) — 15

---

## 🎓 Presentaciones

Se publican en GitHub Pages con cada push a `main`. Orden sugerido (cada una tiene una barra para saltar entre ellas):

1. `mlops.html` — Intro a MLOps (conceptos, herramientas, MLflow, LLMOps)
2. `index.html` — Pipeline práctico (uv → EDA → modelos → CV → MLflow → API)
3. `presentacion_mlops_avanzado.html` — Avanzado (CI/CD → Docker → Monitoreo → Reentrenamiento → Feature Store)

---

## 📊 Stack Tecnológico

| Categoría | Herramientas |
|-----------|-------------|
| **Entorno** | uv, pyproject.toml, uv.lock |
| **Exploración** | Pandas, Seaborn, Matplotlib, Jupyter |
| **Modelado** | Scikit-Learn (10 modelos) |
| **Validación** | Cross-Validation 5-Fold |
| **Tracking** | MLFlow |
| **Almacenamiento** | PyArrow (Parquet) |
| **API** | FastAPI + Uvicorn |
| **Contenedores** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Monitoreo** | Valores atípicos por z-score, registro persistente, cron de salud |
| **Despliegue** | Render / Railway con volumen y cron jobs |

---

<div align="center">

**Universidad EIA** — Ingeniería con propósito 🎓

</div>
