
"""Configuración del proyecto MLOps para predicción de compras Black Friday."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / ".env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "black-friday")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "black-friday-model")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "challenger")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

_ruta_datos = os.getenv("DATA_PATH", "data/train.csv")
DATA_PATH = (
    Path(_ruta_datos)
    if Path(_ruta_datos).is_absolute()
    else RAIZ / _ruta_datos
)

TARGET = "Purchase"
TEST_SIZE = 0.3
METRIC = "rmse"

CATEGORICAS = (
    "Gender",
    "Age",
    "Occupation",
    "City_Category",
    "Stay_In_Current_City_Years",
    "Marital_Status",
    "Product_Category_1",
    "Product_Category_2",
    "Product_Category_3",
)

COLUMNS_TO_DROP = (
    "User_ID",
    "Product_ID",
)


def activar_seguimiento():
    """Configura el almacén y el experimento de MLflow."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
