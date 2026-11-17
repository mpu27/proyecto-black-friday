
"""Entrenamiento de cuatro modelos de regresión para Black Friday."""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import config
from data import cargar_particion, registrar_preprocesamiento


def _modelos():
    semilla = config.RANDOM_SEED

    return [
        (
            "regresion_lineal",
            LinearRegression(),
        ),
        (
            "arbol_decision",
            DecisionTreeRegressor(
                max_depth=20,
                random_state=semilla,
            ),
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=80,
                max_depth=18,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=semilla,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=31,
                random_state=semilla,
            ),
        ),
    ]


def _datasets(x_train, y_train, x_test, y_test):
    origen = str(config.DATA_PATH)

    def conjunto(features, objetivo, nombre):
        tabla = features.copy()
        tabla[config.TARGET] = objetivo

        return mlflow.data.from_pandas(
            tabla,
            source=origen,
            targets=config.TARGET,
            name=nombre,
        )

    return (
        conjunto(x_train, y_train, "black_friday_train"),
        conjunto(x_test, y_test, "black_friday_test"),
    )


def _parametros(modelo):
    parametros = {}

    for clave, valor in modelo.get_params().items():
        if isinstance(valor, (int, float, str, bool)):
            parametros[clave] = valor
        else:
            parametros[clave] = (
                "None" if valor is None else str(valor)
            )

    return parametros


def entrenar():
    os.environ.pop("MLFLOW_RUN_ID", None)

    config.activar_seguimiento()
    mlflow.sklearn.autolog(disable=True)

    (
        x_train,
        x_test,
        y_train,
        y_test,
        _,
        material,
    ) = cargar_particion()

    with mlflow.start_run(run_name="preprocesamiento"):
        registrar_preprocesamiento(material)

    print(
        f"Partición: train {len(x_train)} filas, "
        f"test {len(x_test)} filas, "
        f"{x_train.shape[1]} características"
    )

    datos_train, datos_test = _datasets(
        x_train, y_train, x_test, y_test
    )

    ejemplo = x_train.head(1)

    for nombre, modelo in _modelos():

        with mlflow.start_run(run_name=nombre):

            mlflow.log_input(datos_train, context="training")
            mlflow.log_input(datos_test, context="evaluation")

            modelo.fit(x_train, y_train)

            prediccion = modelo.predict(x_test)

            mae = mean_absolute_error(y_test, prediccion)
            rmse = mean_squared_error(y_test, prediccion) ** 0.5
            r2 = r2_score(y_test, prediccion)

            mlflow.log_params(_parametros(modelo))

            mlflow.log_param("estrategia", nombre)
            mlflow.log_param("test_size", config.TEST_SIZE)
            mlflow.log_param("random_seed", config.RANDOM_SEED)

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(
                modelo,
                name="model",
                input_example=ejemplo,
                serialization_format="pickle",
            )

            print(
                f"{nombre}: "
                f"MAE={mae:.4f} | "
                f"RMSE={rmse:.4f} | "
                f"R2={r2:.4f}"
            )


if __name__ == "__main__":
    entrenar()
