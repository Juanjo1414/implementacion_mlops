"""Entrena candidatos de regresión y publica el elegido para la API.

La selección usa únicamente validación cruzada del conjunto de entrenamiento;
el conjunto de prueba queda reservado para estimar el rendimiento final.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import joblib
import pyarrow as pa
import pyarrow.parquet as pq

# MLflow guarda los modelos con skops (formato seguro, no ejecuta código al cargar).
# skops exige declarar explícitamente los tipos internos de numpy/sklearn que usan los modelos.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "sklearn.neighbors._kd_tree.KDTree",
    "sklearn.metrics._dist_metrics.EuclideanDistance64",
]


def load_data(path):
    """Lee el CSV de viviendas sin alterar sus columnas originales."""
    return pd.read_csv(path)


def preprocess_and_split(df):
    """Separa target, reserva 20 % para prueba y define el preprocesador.

    El preprocesador se ajusta dentro de cada pipeline/fold, evitando que
    medianas, escalas y categorías del conjunto de prueba entren al modelo.
    """
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    # skops serializa listas de nombres de columnas, pero no objetos pd.Index.
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # La mediana completa los dormitorios ausentes antes de escalar.
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Las categorías desconocidas durante inferencia se codifican como ceros.
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # ColumnTransformer conserva una sola receta para entrenamiento e inferencia.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, preprocessor


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    """Compara diez candidatos, registra métricas y guarda el mejor por CV RMSE.

    La métrica de prueba se registra para inspección, pero no participa en
    la elección; usarla para escoger el modelo sesgaría esa evaluación.
    """
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {},
        },
        "Ridge": {
            "model": Ridge(alpha=1.0),
            "params": {"alpha": 1.0},
        },
        "Lasso": {
            "model": Lasso(alpha=1.0, max_iter=5000),
            "params": {"alpha": 1.0, "max_iter": 5000},
        },
        "ElasticNet": {
            "model": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
            "params": {"alpha": 1.0, "l1_ratio": 0.5},
        },
        "DecisionTree": {
            "model": DecisionTreeRegressor(max_depth=15, random_state=42),
            "params": {"max_depth": 15},
        },
        "RandomForest": {
            "model": RandomForestRegressor(n_estimators=100, random_state=42),
            "params": {"n_estimators": 100},
        },
        "ExtraTrees": {
            "model": ExtraTreesRegressor(n_estimators=100, random_state=42),
            "params": {"n_estimators": 100},
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
            ),
            "params": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 5},
        },
        "AdaBoost": {
            "model": AdaBoostRegressor(n_estimators=100, random_state=42),
            "params": {"n_estimators": 100},
        },
        "KNN": {
            "model": KNeighborsRegressor(n_neighbors=5),
            "params": {"n_neighbors": 5},
        },
    }

    mlflow.set_experiment("California_Housing_Prediction")

    best_model = None
    best_cv_rmse = float("inf")
    best_model_name = ""
    cv_folds = 5

    print("=" * 90)
    print(
        f"{'Modelo':<22} {'RMSE Test':>12} {'MAE Test':>12} {'R² Test':>10}"
        f"  {'CV RMSE (μ±σ)':>20}"
    )
    print("=" * 90)

    for name, config in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", config["model"])]
            )

            # Cada fold ajusta su propia copia del preprocesador y del modelo.
            cv_scores_neg_mse = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            cv_rmse_scores = np.sqrt(-cv_scores_neg_mse)
            cv_rmse_mean = cv_rmse_scores.mean()
            cv_rmse_std = cv_rmse_scores.std()

            # Tras CV, se ajusta con todo el entrenamiento y se evalúa en test.
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            print(
                f"{name:<22} {rmse:>12,.2f} {mae:>12,.2f} {r2:>10.4f}"
                f"  {cv_rmse_mean:>9,.2f} ± {cv_rmse_std:>7,.2f}"
            )

            # Los parámetros permiten reconstruir cada candidato desde MLflow.
            mlflow.log_param("model_type", name)
            mlflow.log_param("cv_folds", cv_folds)
            for param_name, param_value in config["params"].items():
                mlflow.log_param(param_name, param_value)

            # Estas métricas describen generalización, no el criterio de elección.
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mse", mse)

            # Se guardan promedio, dispersión y resultados por fold.
            mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
            mlflow.log_metric("cv_rmse_std", cv_rmse_std)
            for i, fold_rmse in enumerate(cv_rmse_scores):
                mlflow.log_metric(f"cv_rmse_fold_{i+1}", fold_rmse)

            # skops necesita la lista explícita de tipos confiables anterior.
            mlflow.sklearn.log_model(
                pipeline, name="model", skops_trusted_types=SKOPS_TRUSTED_TYPES
            )

            if cv_rmse_mean < best_cv_rmse:
                best_cv_rmse = cv_rmse_mean
                best_model = pipeline
                best_model_name = name

    print("=" * 90)
    print(f"\n🏆 Mejor modelo: {best_model_name} con CV RMSE: {best_cv_rmse:,.2f}\n")

    # Guardar el mejor modelo para la API
    joblib.dump(best_model, "best_model.pkl")

    # Parquet conserva entradas, valores reales y predicciones para análisis.
    test_results = X_test.copy()
    test_results["actual_median_house_value"] = y_test
    test_results["predicted_median_house_value"] = best_model.predict(X_test)

    table = pa.Table.from_pandas(test_results)
    pq.write_table(table, "test_predictions.parquet")
    print("Resultados guardados en test_predictions.parquet")
    print(
        "\n📊 Para ver los experimentos en MLFlow ejecuta:\n"
        "   uv run mlflow ui --port 5001\n"
        "   Luego abre http://localhost:5001 en tu navegador\n"
    )


if __name__ == "__main__":
    df = load_data("1553768847-housing.csv")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)
    train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
