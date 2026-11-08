"""Leitura da base, pré-processamento e reconstrução dos centroides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    CHAS_CATEGORIES,
    DATA_DIR,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)


class DataValidationError(ValueError):
    """A base de dados não atende ao formato esperado pelo projeto."""


def validate_columns(df: pd.DataFrame) -> None:
    """Confere colunas obrigatórias, tipos numéricos, infinitos e categorias de CHAS.

    MEDV não é obrigatória, pois não participa da modelagem. Valores ausentes são
    aceitos aqui e tratados pelos imputadores do pipeline.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise DataValidationError(f"Colunas obrigatórias ausentes na base: {missing}")

    non_numeric = [
        column for column in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise DataValidationError(f"Colunas com valores não numéricos: {non_numeric}")

    infinite = [column for column in FEATURE_COLUMNS if np.isinf(df[column].to_numpy()).any()]
    if infinite:
        raise DataValidationError(f"Colunas com valores infinitos: {infinite}")

    invalid_chas = sorted(set(df["CHAS"].dropna().unique()) - set(CHAS_CATEGORIES))
    if invalid_chas:
        raise DataValidationError(f"CHAS deve conter apenas 0 ou 1; encontrado: {invalid_chas}")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Lê o CSV preservando todos os registros; `NA` vira valor ausente."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Base não encontrada em '{path}'. Copie o arquivo HousingData.csv para "
            f"'{DATA_DIR.name}/' ou informe outro caminho com --data."
        )
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DataValidationError(f"Não foi possível ler '{path}' como CSV: {error}") from error

    validate_columns(df)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna somente as 13 características, na ordem canônica e sem MEDV."""
    return df.loc[:, FEATURE_COLUMNS].copy()


def build_preprocessor() -> ColumnTransformer:
    """Cria o ColumnTransformer ainda não ajustado.

    Saída: 12 colunas numéricas padronizadas seguidas de 2 colunas One-Hot de CHAS.
    As colunas One-Hot não recebem StandardScaler, de modo que a média de cada uma
    em um cluster é diretamente a proporção da categoria.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    categories=[CHAS_CATEGORIES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def imputation_values(pipeline: Pipeline) -> dict[str, float]:
    """Retorna, por variável, o valor que os imputadores ajustados usam para ausentes."""
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    categorical_imputer = preprocessor.named_transformers_["categorical"].named_steps["imputer"]
    values = dict(zip(NUMERIC_FEATURES, numeric_imputer.statistics_))
    values.update(zip(CATEGORICAL_FEATURES, categorical_imputer.statistics_))
    return {feature: float(value) for feature, value in values.items()}


def denormalize_centroids(pipeline: Pipeline) -> list[dict[str, Any]]:
    """Reconstrói os centroides na escala original a partir dos transformadores ajustados.

    O `inverse_transform` genérico não é usado porque o ColumnTransformer não o oferece.
    Cada bloco é tratado de acordo com seu significado:

    - numéricas: `StandardScaler.inverse_transform` sobre as colunas numéricas. O
      resultado é a média do cluster (com valores imputados), e não um registro real;
    - CHAS: as colunas One-Hot não foram padronizadas, então o centroide já contém a
      proporção de cada categoria no cluster.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    centers = pipeline.named_steps["kmeans"].cluster_centers_

    numeric_slice = preprocessor.output_indices_["numeric"]
    categorical_slice = preprocessor.output_indices_["categorical"]

    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]

    # O arredondamento elimina resíduos de ponto flutuante como -1e-14 ("-0.0000"), que
    # sugeririam valores negativos impossíveis em variáveis como ZN.
    numeric_values = np.round(scaler.inverse_transform(centers[:, numeric_slice]), 10) + 0.0
    proportions = centers[:, categorical_slice]
    categories = [int(category) for category in encoder.categories_[0]]

    profiles = []
    for cluster, (numeric_row, proportion_row) in enumerate(zip(numeric_values, proportions)):
        dominant = int(np.argmax(proportion_row))
        profiles.append(
            {
                "cluster": cluster,
                "numeric": dict(zip(NUMERIC_FEATURES, map(float, numeric_row))),
                "CHAS": {
                    "proporcoes": dict(zip(categories, map(float, proportion_row))),
                    "predominante": categories[dominant],
                    "predominante_pct": float(proportion_row[dominant] * 100),
                },
            }
        )
    return profiles
