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
├── train.py                    # 🤖 10 modelos; selección por CV 5-Fold
├── app.py                      # 🚀 API REST con FastAPI
├── pycaret_app.py              # 🚀 API del pipeline exportado por PyCaret
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
├── compose.yml                 # 🐳 Docker Compose (API + MLFlow)
├── model_metrics.json          # 📋 Métricas del modelo actual
├── tests/
│   ├── test_pipeline.py        # 🧪 Pipeline original
│   └── test_pycaret_serving.py # 🧪 Esquema y API PyCaret
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

El monitor no dispara el reentrenamiento ni mide error de predicción, porque la API no recibe el valor real de la vivienda. El workflow semanal ejecuta `retrain.py` y publica `best_model.pkl` junto con sus métricas como artefacto de GitHub Actions; no despliega el modelo en la API. Para usarlo localmente, descarga el artefacto y reinicia la API. La comparación de RMSE en `retrain.py` usa el holdout del CSV y solo es comparable si los datos y la partición de evaluación siguen siendo los mismos.

---

## 🐳 Docker

```bash
# Construir y levantar MLflow, Jupyter y la API PyCaret
docker compose up --build

# El token de Jupyter aparece en los logs del servicio
docker compose logs notebook

# Después de ejecutar pycaret_experimentos.ipynb
docker compose restart api
```

Abre Jupyter en `http://localhost:8888`, MLflow en `http://localhost:5050` y Swagger de FastAPI en `http://localhost:8000/docs`. Las tres imágenes se construyen con `uv.lock`: API y MLflow usan dependencias principales; Jupyter añade los grupos `eda` y `pycaret`. El directorio `models/` se comparte entre notebook y API, y un volumen nombrado conserva el servidor MLflow. Hasta exportar el modelo, `/health` responde 503; después de exportarlo hay que reiniciar la API para que cargue el nuevo archivo.

---

## 🧪 Tests

```bash
uv run python train.py        # los tests necesitan best_model.pkl
uv run pytest -v
uv run flake8 *.py tests/ --max-line-length=120
```

23 tests que cubren:
- ✅ Feature Store (features derivadas, preprocesador, info, versiones) — 4
- ✅ Monitor (logging, drift normal, drift extremo, resumen) — 5
- ✅ Integridad de datos (CSV existe, columnas correctas, modelo existe) — 3
- ✅ API (`/health`, `/predict` con el modelo de `train.py` y el de `retrain.py`, modelo ausente → 503, payload inválido → 422, valores atípicos) — 6
- ✅ Serialización (los modelos se pueden guardar en MLflow con skops) — 2
- ✅ PyCaret (esquema de features, inferencia y errores de la API) — 3

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
| **Monitoreo** | Valores atípicos por z-score |

---

<div align="center">

**Universidad EIA** — Ingeniería con propósito 🎓

</div>
