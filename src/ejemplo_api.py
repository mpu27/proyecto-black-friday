
"""Genera un ejemplo de petición y respuesta para la API Black Friday."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

from mlflow.models import Model
from mlflow.tracking import MlflowClient

import config


SALIDA = config.RAIZ / "ejemplos-api"
PUERTO = 5001


def _uri(modelo: str, alias: str) -> str:
    return f"models:/{modelo}@{alias}"


def _jsonable(valor):
    if isinstance(valor, dict):
        return {
            clave: _jsonable(item)
            for clave, item in valor.items()
        }

    if isinstance(valor, (list, tuple)):
        return [_jsonable(item) for item in valor]

    if hasattr(valor, "item"):
        return valor.item()

    return valor


def _payload(ejemplo):
    if ejemplo is None:
        return None

    if isinstance(ejemplo, pd.DataFrame):
        return {
            "dataframe_split": {
                "columns": list(ejemplo.columns),
                "data": _jsonable(ejemplo.to_numpy().tolist()),
            }
        }

    return _jsonable(ejemplo)


def _predecir(uri: str, ejemplo):
    if not isinstance(ejemplo, pd.DataFrame):
        return None

    modelo = mlflow.pyfunc.load_model(uri)

    prediccion = modelo.predict(ejemplo)

    if hasattr(prediccion, "tolist"):
        prediccion = prediccion.tolist()

    return {
        "predictions": _jsonable(prediccion)
    }


def _esquema(firma, lado: str) -> str:
    tensor = (
        getattr(firma, lado, None)
        if firma is not None
        else None
    )

    if tensor is None:
        return "No hay firma registrada."

    return json.dumps(
        tensor.to_dict(),
        indent=2,
        ensure_ascii=False,
    )


def generar(
    modelo: str,
    alias: str,
    puerto: int = PUERTO
) -> Path:

    config.activar_seguimiento()

    uri = _uri(modelo, alias)

    cliente = MlflowClient(
        tracking_uri=config.MLFLOW_TRACKING_URI
    )

    version = cliente.get_model_version_by_alias(
        modelo,
        alias
    )

    info = mlflow.models.get_model_info(uri)

    ejemplo = Model.load(uri).load_input_example(uri)

    if isinstance(ejemplo, pd.DataFrame):
        ejemplo = ejemplo.head(1)

    payload = _payload(ejemplo)

    respuesta = (
        _predecir(uri, ejemplo)
        if payload is not None
        else None
    )

    cuerpo_peticion = (
        json.dumps(payload, indent=2, ensure_ascii=False)
        if payload is not None
        else None
    )

    cuerpo_respuesta = (
        json.dumps(respuesta, indent=2, ensure_ascii=False)
        if respuesta is not None
        else None
    )

    curl = ""

    if cuerpo_peticion is not None:
        curl = (
            f"curl http://127.0.0.1:{puerto}/invocations \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            f"  -d '{json.dumps(payload, ensure_ascii=False)}'"
        )

    lineas = [
        f"# {modelo} @ {alias}",
        "",
        f"- URI: `{uri}`",
        f"- Versión: {version.version}",
        f"- Tipo de problema: Regresión",
        f"- Variable objetivo: `{config.TARGET}`",
        f"- Endpoint: `POST http://127.0.0.1:{puerto}/invocations`",
        "",
        "## Esquema de entrada",
        "",
        "```json",
        _esquema(info.signature, "inputs"),
        "```",
        "",
        "## Esquema de salida",
        "",
        "```json",
        _esquema(info.signature, "outputs"),
        "```",
        "",
        "## Petición",
        "",
    ]

    if cuerpo_peticion is None:
        lineas.append(
            "Este modelo no tiene input_example registrado."
        )
    else:
        lineas.extend([
            "```bash",
            curl,
            "```",
            "",
            "```json",
            cuerpo_peticion,
            "```",
        ])

    lineas.extend([
        "",
        "## Respuesta",
        "",
    ])

    if cuerpo_respuesta is None:
        lineas.append(
            "No se pudo generar la predicción."
        )
    else:
        lineas.extend([
            "```json",
            cuerpo_respuesta,
            "```",
        ])

    lineas.append("")

    SALIDA.mkdir(parents=True, exist_ok=True)

    destino = SALIDA / f"{modelo}__{alias}.md"

    destino.write_text(
        "\n".join(lineas),
        encoding="utf-8"
    )

    return destino


def main():

    parser = argparse.ArgumentParser(
        description="Genera ejemplos de la API del modelo Black Friday."
    )

    parser.add_argument(
        "--modelo",
        default=config.MLFLOW_MODEL_NAME,
    )

    parser.add_argument(
        "--alias",
        action="append",
        default=None,
    )

    parser.add_argument(
        "--puerto",
        type=int,
        default=PUERTO,
    )

    args = parser.parse_args()

    alias = args.alias or [config.MLFLOW_MODEL_ALIAS]

    for nombre in alias:
        destino = generar(
            args.modelo,
            nombre,
            args.puerto
        )

        print(f"Documento generado: {destino}")


if __name__ == "__main__":
    main()
