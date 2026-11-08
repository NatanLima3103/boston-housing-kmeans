"""Inferência: classifica uma nova observação com o artefato salvo, sem novo treinamento."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import (
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PERCENT_FEATURES,
    TARGET_COLUMN,
    configure_console,
)
from src.preprocessing import denormalize_centroids, imputation_values

ID_FIELD = "identificacao"
NULL_TOKENS = {"null", "na", "nulo"}
LINE_WIDTH = 46
DEFAULT_ID = "Não informada"


class InvalidObservationError(ValueError):
    """A observação informada não pode ser classificada."""


class ModelLoadError(RuntimeError):
    """O artefato do modelo não existe ou não pode ser utilizado."""


@dataclass
class Observation:
    """Observação validada: valores nas 13 características, com NaN para `null`."""

    identificacao: str
    values: dict[str, float]
    warnings: list[str] = field(default_factory=list)


def load_artifact(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Carrega o artefato Joblib e confere sua estrutura.

    Joblib usa pickle, que pode executar código ao carregar: use apenas artefatos de
    origem confiável, como os gerados por `python -m src.train`.
    """
    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(
            f"Modelo não encontrado em '{path}'. Execute primeiro: python -m src.train"
        )
    try:
        artifact = joblib.load(path)
    except Exception as error:  # joblib/pickle podem levantar exceções de vários tipos
        # A exceção original fica encadeada (__cause__) para depuração, mas não é exibida:
        # mensagens internas do pickle, como "pop from empty list", não orientam o usuário.
        raise ModelLoadError(
            f"Não foi possível carregar o modelo '{path}'. O arquivo pode estar corrompido, "
            "incompleto ou ter sido gerado com versões de bibliotecas diferentes das de "
            "requirements.txt. Verifique o arquivo ou gere-o novamente com: python -m src.train"
        ) from error

    if not isinstance(artifact, dict) or not {"pipeline", "metadata"} <= artifact.keys():
        raise ModelLoadError(f"O arquivo '{path}' não é um artefato válido deste projeto.")
    named_steps = getattr(artifact["pipeline"], "named_steps", None)
    if named_steps is None or not {"preprocessor", "kmeans"} <= named_steps.keys():
        raise ModelLoadError(f"O pipeline em '{path}' não contém as etapas esperadas.")
    return artifact


def _reject_json_constant(token: str) -> None:
    raise InvalidObservationError(f"Valor '{token}' não é permitido: use um número finito.")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    duplicated = sorted({key for key in keys if keys.count(key) > 1})
    if duplicated:
        raise InvalidObservationError(f"Campos duplicados no JSON: {duplicated}")
    return dict(pairs)


def read_json_observation(path: str | Path) -> dict[str, Any]:
    """Lê o JSON de entrada; NaN e Infinity, aceitos pelo módulo json, são rejeitados."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise InvalidObservationError(f"Não foi possível ler '{path}': {error}") from error
    except UnicodeDecodeError as error:
        raise InvalidObservationError(f"O arquivo '{path}' não está em UTF-8.") from error

    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise InvalidObservationError(
            f"JSON inválido em '{path}' (linha {error.lineno}, coluna {error.colno}): {error.msg}"
        ) from error


def validate_value(feature: str, value: Any) -> float:
    """Valida um valor e o converte para float; `None` vira NaN, a ser imputado."""
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidObservationError(
            f"{feature}: esperado um número ou null, recebido {type(value).__name__}."
        )
    try:
        number = float(value)
    except OverflowError as error:
        raise InvalidObservationError(f"{feature}: número grande demais.") from error
    if not math.isfinite(number):
        raise InvalidObservationError(f"{feature}: o valor deve ser finito.")
    if feature == "CHAS":
        if number not in (0.0, 1.0):
            raise InvalidObservationError(f"CHAS: deve ser 0 ou 1, recebido {value}.")
        return number
    if number < 0:
        raise InvalidObservationError(f"{feature}: não pode ser negativo por definição.")
    if feature in PERCENT_FEATURES and number > 100:
        raise InvalidObservationError(f"{feature}: é um percentual e não pode passar de 100.")
    return number


def validate_observation(raw: Any) -> Observation:
    """Valida o registro completo e reúne todos os problemas em uma única mensagem."""
    if not isinstance(raw, dict):
        raise InvalidObservationError("A entrada deve ser um objeto JSON com as 13 características.")

    problems: list[str] = []
    if TARGET_COLUMN in raw:
        problems.append(f"{TARGET_COLUMN} não faz parte do modelo e não deve ser informada.")

    allowed = set(FEATURE_COLUMNS) | {ID_FIELD, TARGET_COLUMN}
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        problems.append(f"Características inesperadas: {unexpected}")

    absent = [feature for feature in FEATURE_COLUMNS if feature not in raw]
    if absent:
        problems.append(f"Campos obrigatórios ausentes: {absent}")

    identificacao = raw.get(ID_FIELD)
    if identificacao is not None and not isinstance(identificacao, str):
        problems.append(f"{ID_FIELD}: deve ser um texto.")

    values: dict[str, float] = {}
    for feature in FEATURE_COLUMNS:
        if feature in raw:
            try:
                values[feature] = validate_value(feature, raw[feature])
            except InvalidObservationError as error:
                problems.append(str(error))

    if problems:
        raise InvalidObservationError("\n".join(f"- {problem}" for problem in problems))

    label = identificacao.strip() if isinstance(identificacao, str) else ""
    return Observation(identificacao=label or DEFAULT_ID, values=values)


def parse_interactive_value(text: str) -> Any:
    """Converte o texto digitado; aceita vírgula decimal e `null` para valor ausente."""
    cleaned = text.strip()
    if cleaned.lower() in NULL_TOKENS:
        return None
    if not cleaned:
        raise InvalidObservationError("Informe um número, ou null se o valor for desconhecido.")
    try:
        return float(cleaned.replace(",", "."))
    except ValueError as error:
        raise InvalidObservationError(f"'{cleaned}' não é um número válido.") from error


def collect_interactive(prompt: Callable[[str], str] = input) -> dict[str, Any]:
    """Solicita os dados pelo terminal, repetindo a pergunta quando o valor é inválido."""
    print("Nova residência (tecnicamente, um novo perfil de região censitária)")
    print("Digite null para um valor desconhecido; ele será imputado pelo modelo.\n")
    raw: dict[str, Any] = {ID_FIELD: prompt("Identificação (opcional): ").strip() or None}
    for feature in FEATURE_COLUMNS:
        while True:
            try:
                value = parse_interactive_value(
                    prompt(f"{feature} - {FEATURE_DESCRIPTIONS[feature]}: ")
                )
                validate_value(feature, value)
            except InvalidObservationError as error:
                print(f"  Valor inválido. {error}")
                continue
            raw[feature] = value
            break
    return raw


def build_warnings(observation: Observation, artifact: dict[str, Any]) -> list[str]:
    """Gera avisos informativos: imputações, valores atípicos e versão do scikit-learn."""
    metadata = artifact["metadata"]
    imputed = imputation_values(artifact["pipeline"])
    warnings = []

    if metadata["sklearn_version"] != sklearn.__version__:
        warnings.append(
            f"Modelo treinado com scikit-learn {metadata['sklearn_version']}; "
            f"versão instalada: {sklearn.__version__}."
        )

    for feature, value in observation.values.items():
        if math.isnan(value):
            rule = "moda" if feature == "CHAS" else "mediana"
            warnings.append(
                f"{feature} ausente (null): imputado pela {rule} do treinamento ({imputed[feature]:g})."
            )
        elif feature in metadata["train_ranges"]:
            limits = metadata["train_ranges"][feature]
            if not limits["min"] <= value <= limits["max"]:
                warnings.append(
                    f"{feature} = {value:g} está fora do intervalo do treinamento "
                    f"[{limits['min']:g}, {limits['max']:g}]."
                )

    rad = observation.values["RAD"]
    if not math.isnan(rad) and rad not in metadata["rad_observed_values"]:
        warnings.append(f"RAD = {rad:g} não foi observado como índice no treinamento.")
    return warnings


def predict_observation(observation: Observation, artifact: dict[str, Any]) -> dict[str, Any]:
    """Classifica a observação reutilizando os parâmetros aprendidos no treinamento."""
    pipeline = artifact["pipeline"]
    metadata = artifact["metadata"]
    frame = pd.DataFrame([observation.values], columns=FEATURE_COLUMNS, dtype=float)

    cluster = int(pipeline.predict(frame)[0])
    transformed = pipeline.named_steps["preprocessor"].transform(frame)
    center = pipeline.named_steps["kmeans"].cluster_centers_[cluster]
    distance = float(np.linalg.norm(transformed[0] - center))

    return {
        "identificacao": observation.identificacao,
        "cluster": cluster,
        "distancia": distance,
        "distancia_cluster": metadata["distance_by_cluster"][cluster],
        "centroide": denormalize_centroids(pipeline)[cluster],
        "rad_mais_frequente": metadata["rad_mode_by_cluster"][cluster],
        "avisos": build_warnings(observation, artifact),
    }


def format_result(result: dict[str, Any]) -> str:
    """Monta o texto do resultado apresentado no terminal."""
    heavy = "=" * LINE_WIDTH
    light = "-" * LINE_WIDTH
    centroid = result["centroide"]
    chas = centroid["CHAS"]
    reference = result["distancia_cluster"]
    rad_mode = result["rad_mais_frequente"]

    lines = [
        heavy,
        "RESULTADO DA INFERÊNCIA".center(LINE_WIDTH),
        heavy,
        "",
        f"Identificação: {result['identificacao']}",
        "",
        f"Cluster atribuído: {result['cluster']}",
        "(O número identifica o grupo; não indica qualidade)",
        "",
        f"Distância até o centroide: {result['distancia']:.4f}",
        "(Espaço normalizado e codificado)",
        f"Referência do cluster no treinamento: média {reference['distancia_media']:.4f}, "
        f"p95 {reference['distancia_p95']:.4f}",
        "",
        light,
        "CENTROIDE NA ESCALA ORIGINAL",
        light,
        "(Perfil médio do cluster, não uma região real)",
        "",
    ]
    lines += [f"{name + ':':<12}{centroid['numeric'][name]:.4f}" for name in NUMERIC_FEATURES]
    lines += ["", "CHAS:", ""]
    lines += [
        f"Categoria {category}: {proportion * 100:.2f}%"
        for category, proportion in chas["proporcoes"].items()
    ]
    lines += [
        "",
        f"Categoria predominante: {chas['predominante']} ({chas['predominante_pct']:.2f}%)",
        "",
        light,
        "ESTATÍSTICA COMPLEMENTAR (NÃO FAZ PARTE DO CENTROIDE)",
        light,
        f"RAD mais frequente nos registros reais do cluster: {rad_mode['valor']:g} "
        f"({rad_mode['pct']:.2f}%)",
    ]
    if result["avisos"]:
        lines += ["", light, "AVISOS", light]
        lines += [f"- {warning}" for warning in result["avisos"]]
    lines += ["", heavy]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada de `python -m src.inference`."""
    configure_console()
    parser = argparse.ArgumentParser(
        description="Classifica uma nova observação com o modelo K-Means salvo."
    )
    parser.add_argument("--json", type=Path, default=None, help="Arquivo JSON com a observação.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Caminho do artefato Joblib.")
    args = parser.parse_args(argv)

    try:
        artifact = load_artifact(args.model)
        raw = read_json_observation(args.json) if args.json else collect_interactive()
        observation = validate_observation(raw)
        print(format_result(predict_observation(observation, artifact)))
    except ModelLoadError as error:
        print(f"Erro no modelo: {error}", file=sys.stderr)
        return 1
    except InvalidObservationError as error:
        print(f"Entrada inválida:\n{error}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nEntrada cancelada.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
