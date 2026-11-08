"""Configuração central do projeto: caminhos, variáveis e hiperparâmetros."""

from __future__ import annotations

import sys
from pathlib import Path

# Caminhos sempre relativos à raiz do projeto, nunca absolutos.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_PATH = DATA_DIR / "HousingData.csv"
MODEL_PATH = MODELS_DIR / "kmeans_model.joblib"
# Treinamentos com K diferente de FINAL_K vão para experiments/k<K>/, preservando os
# arquivos oficiais de models/ e reports/.
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# MEDV é excluída de toda a modelagem; só aparece em análises descritivas a posteriori.
TARGET_COLUMN = "MEDV"

CATEGORICAL_FEATURES = ["CHAS"]
NUMERIC_FEATURES = [
    "CRIM",
    "ZN",
    "INDUS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]

# Ordem canônica das 13 características de entrada (mesma ordem do CSV).
FEATURE_COLUMNS = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]

# Categorias explícitas de CHAS: garantem as duas colunas One-Hot em qualquer base.
CHAS_CATEGORIES = [0.0, 1.0]

# Hiperparâmetros do K-Means.
RANDOM_STATE = 42
N_INIT = 20
MAX_ITER = 300
ALGORITHM = "lloyd"

# Faixa avaliada na seleção de K.
K_MIN = 1
K_MAX = 10

# Menor participação aceitável de um cluster na seleção automática de K, em %.
MIN_CLUSTER_PCT = 5.0

# Decisão final do responsável pelo projeto, justificada no README. A regra automática
# continua sendo calculada e registrada em reports/k_selection.json. Use None para que
# ela volte a definir K, por exemplo ao treinar com outra versão da base.
FINAL_K: int | None = 2

# Reamostragens usadas para medir a estabilidade de cada K.
N_BOOTSTRAP = 50

FEATURE_DESCRIPTIONS = {
    "CRIM": "Taxa de criminalidade per capita",
    "ZN": "% de terrenos residenciais zoneados para lotes acima de 25.000 pés²",
    "INDUS": "% de área comercial não varejista",
    "CHAS": "Fronteira com o rio Charles (0 = não, 1 = sim)",
    "NOX": "Concentração de óxidos nítricos (partes por 10 milhões)",
    "RM": "Número médio de cômodos por residência",
    "AGE": "% de unidades ocupadas pelos proprietários construídas antes de 1940",
    "DIS": "Distância ponderada até cinco centros de emprego de Boston",
    "RAD": "Índice de acessibilidade a rodovias radiais",
    "TAX": "Imposto sobre propriedade por US$ 10.000",
    "PTRATIO": "Razão aluno-professor",
    "B": "1000(Bk - 0,63)², em que Bk é a proporção de pessoas negras (variável histórica)",
    "LSTAT": "% da população de menor status socioeconômico",
}

# Limites decorrentes da definição das variáveis: taxas, distâncias e contagens não
# são negativas, e percentuais não passam de 100. Valores dentro destes limites, mas
# fora do intervalo do treinamento, geram apenas aviso informativo.
PERCENT_FEATURES = ["ZN", "INDUS", "AGE", "LSTAT"]


def configure_console() -> None:
    """Usa UTF-8 na saída, evitando acentos corrompidos ao redirecionar no Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
