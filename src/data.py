
"""Preparación de datos para el proyecto Black Friday."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import config

MUESTRA = 10


def _leer_csv() -> pd.DataFrame:
    if not config.DATA_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset: {config.DATA_PATH}"
        )

    datos = pd.read_csv(config.DATA_PATH)

    requeridas = [config.TARGET, *config.CATEGORICAS]
    faltan = [c for c in requeridas if c not in datos.columns]

    if faltan:
        raise ValueError(f"Faltan columnas en el CSV: {faltan}")

    return datos


def _limpiar(datos: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    resumen = {
        "archivo": config.DATA_PATH.name,
        "filas_leidas": int(len(datos)),
    }

    n = len(datos)
    datos = datos.drop_duplicates()
    resumen["filas_duplicadas"] = int(n - len(datos))

    n = len(datos)
    datos = datos.dropna(subset=[config.TARGET])
    resumen["filas_sin_objetivo"] = int(n - len(datos))

    datos[config.TARGET] = pd.to_numeric(
        datos[config.TARGET], errors="coerce"
    )
    datos = datos.dropna(subset=[config.TARGET])

    columnas_eliminar = [
        c for c in config.COLUMNS_TO_DROP if c in datos.columns
    ]
    datos = datos.drop(columns=columnas_eliminar)

    resumen["columnas_eliminadas"] = columnas_eliminar

    # Las variables categóricas faltantes se identifican como Sin_dato.
    for columna in config.CATEGORICAS:
        datos[columna] = datos[columna].fillna("Sin_dato").astype(str)

    datos = datos.reset_index(drop=True)

    resumen["filas_limpias"] = int(len(datos))

    return datos, resumen


def cargar_limpio() -> pd.DataFrame:
    datos, _ = _limpiar(_leer_csv())
    return datos


def ajustar_preprocesador(datos: pd.DataFrame) -> dict:
    """Ajusta codificación y escalado."""

    trabajo = datos.copy()

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    trabajo[list(config.CATEGORICAS)] = encoder.fit_transform(
        trabajo[list(config.CATEGORICAS)]
    )

    features = list(config.CATEGORICAS)

    scaler = StandardScaler()
    scaler.fit(trabajo[features])

    return {
        "encoder": encoder,
        "scaler": scaler,
        "features": features,
    }


def aplicar_preprocesador(
    datos: pd.DataFrame,
    preprocesador: dict
) -> pd.DataFrame:

    trabajo = datos.copy()

    if config.TARGET in trabajo.columns:
        trabajo = trabajo.drop(columns=[config.TARGET])

    trabajo = trabajo[list(config.CATEGORICAS)].copy()

    for columna in config.CATEGORICAS:
        trabajo[columna] = (
            trabajo[columna].fillna("Sin_dato").astype(str)
        )

    trabajo[list(config.CATEGORICAS)] = (
        preprocesador["encoder"].transform(
            trabajo[list(config.CATEGORICAS)]
        )
    )

    escaladas = preprocesador["scaler"].transform(
        trabajo[preprocesador["features"]]
    )

    return pd.DataFrame(
        escaladas,
        columns=preprocesador["features"],
        index=trabajo.index,
    )


def _transformaciones(preprocesador: dict) -> dict:
    encoder = preprocesador["encoder"]

    codificacion = {
        columna: {
            str(valor): int(indice)
            for indice, valor in enumerate(categorias)
        }
        for columna, categorias in zip(
            config.CATEGORICAS, encoder.categories_
        )
    }

    scaler = preprocesador["scaler"]

    escalado = {
        columna: {
            "media": float(scaler.mean_[i]),
            "escala": float(scaler.scale_[i]),
        }
        for i, columna in enumerate(preprocesador["features"])
    }

    return {
        "objetivo": config.TARGET,
        "test_size": config.TEST_SIZE,
        "random_seed": config.RANDOM_SEED,
        "metric": config.METRIC,
        "codificacion": codificacion,
        "escalado": escalado,
        "features": preprocesador["features"],
    }


def cargar_particion():
    crudo = _leer_csv()
    datos, resumen = _limpiar(crudo)

    preprocesador = ajustar_preprocesador(datos)

    features = aplicar_preprocesador(datos, preprocesador)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        datos[config.TARGET].astype(float),
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )

    material = {
        "resumen": resumen,
        "transformaciones": _transformaciones(preprocesador),
        "muestra_cruda": crudo.head(MUESTRA),
        "muestra_limpia": datos.head(MUESTRA),
        "muestra_transformada": features.head(MUESTRA),
        "filas_train": len(x_train),
        "filas_test": len(x_test),
    }

    return (
        x_train,
        x_test,
        y_train,
        y_test,
        preprocesador,
        material,
    )


def registrar_preprocesamiento(material: dict) -> None:
    import mlflow

    if mlflow.active_run() is None:
        raise RuntimeError("Se necesita un run activo de MLflow.")

    resumen = material["resumen"]

    mlflow.log_params({
        "archivo": resumen["archivo"],
        "filas_leidas": resumen["filas_leidas"],
        "filas_duplicadas": resumen["filas_duplicadas"],
        "filas_sin_objetivo": resumen["filas_sin_objetivo"],
        "filas_limpias": resumen["filas_limpias"],
        "filas_train": material["filas_train"],
        "filas_test": material["filas_test"],
        "objetivo": config.TARGET,
        "columnas_categoricas": ",".join(config.CATEGORICAS),
    })

    mlflow.log_dict(
        {"limpieza": resumen, **material["transformaciones"]},
        "preprocesamiento/transformaciones.json",
    )

    mlflow.log_table(
        material["muestra_cruda"],
        "preprocesamiento/muestra_cruda.json",
    )

    mlflow.log_table(
        material["muestra_limpia"],
        "preprocesamiento/muestra_limpia.json",
    )

    mlflow.log_table(
        material["muestra_transformada"],
        "preprocesamiento/muestra_transformada.json",
    )


def fila_cruda_ejemplo() -> pd.DataFrame:
    datos = cargar_limpio()
    return datos.drop(columns=[config.TARGET]).head(1)
