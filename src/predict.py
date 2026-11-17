
"""Carga el modelo registrado y predice compras Black Friday."""

from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import Model

import config


def _uri() -> str:
    return (
        f"models:/{config.MLFLOW_MODEL_NAME}"
        f"@{config.MLFLOW_MODEL_ALIAS}"
    )


def cargar():
    """Carga el modelo de regresión mediante su alias."""

    config.activar_seguimiento()

    return mlflow.sklearn.load_model(_uri())


def predecir(datos: pd.DataFrame):
    """Predice el valor de compra de una o varias observaciones."""

    return cargar().predict(datos)


def main():

    parser = argparse.ArgumentParser(
        description="Predice el valor de compra en Black Friday."
    )

    parser.add_argument(
        "--input",
        default=None,
        help="CSV con características previamente transformadas."
    )

    args = parser.parse_args()

    config.activar_seguimiento()

    if args.input:
        datos = pd.read_csv(args.input)
    else:
        datos = Model.load(_uri()).load_input_example(_uri())

        if isinstance(datos, pd.DataFrame):
            datos = datos.head(1)

    if datos is None:
        raise SystemExit(
            "El modelo no tiene input_example. "
            "Proporciona un CSV con --input."
        )

    predicciones = predecir(datos)

    print("Predicciones de compra:")

    for i, valor in enumerate(predicciones, start=1):
        print(f"Registro {i}: {valor:.2f}")


if __name__ == "__main__":
    main()
