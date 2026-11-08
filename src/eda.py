"""Análise exploratória: recalcula as estatísticas a partir do arquivo e gera os relatórios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    TARGET_COLUMN,
    configure_console,
)
from src.preprocessing import DataValidationError, load_dataset

COLORS = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "neutral": "#f0efec",
    "primary": "#2a78d6",
    "accent": "#eb6834",
    "negative": "#e34948",
}

CORRELATION_CMAP = LinearSegmentedColormap.from_list(
    "correlacao", [COLORS["negative"], COLORS["neutral"], COLORS["primary"]]
)


def apply_plot_style() -> None:
    """Aplica o estilo comum a todos os gráficos do projeto."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["surface"],
            "axes.facecolor": COLORS["surface"],
            "savefig.facecolor": COLORS["surface"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink_secondary"],
            "axes.edgecolor": COLORS["axis"],
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "legend.frameon": False,
        }
    )


def _analysis_columns(df: pd.DataFrame) -> list[str]:
    """Colunas analisadas: as 13 características e MEDV, se existir no arquivo."""
    target = [TARGET_COLUMN] if TARGET_COLUMN in df.columns else []
    return FEATURE_COLUMNS + target


def _quantitative_columns(df: pd.DataFrame) -> list[str]:
    """Variáveis quantitativas, sem CHAS, acrescidas de MEDV para fins descritivos."""
    target = [TARGET_COLUMN] if TARGET_COLUMN in df.columns else []
    return NUMERIC_FEATURES + target


def _iqr_limits(values: pd.DataFrame | pd.Series) -> tuple:
    """Limites inferior e superior do critério de 1,5 × IQR."""
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Resume dimensões, ausências, duplicidades e a composição de CHAS."""
    analysis = df[_analysis_columns(df)]
    missing_per_record = analysis.isna().sum(axis=1)
    records_with_missing = int((missing_per_record > 0).sum())
    total_cells = analysis.size
    total_missing = int(analysis.isna().sum().sum())

    numeric = df[NUMERIC_FEATURES]
    lower, upper = _iqr_limits(numeric)
    records_with_outlier = int(((numeric < lower) | (numeric > upper)).any(axis=1).sum())

    rows: list[tuple[str, float]] = [
        ("registros", len(df)),
        ("colunas", df.shape[1]),
        ("celulas_ausentes", total_missing),
        ("pct_celulas_ausentes", round(100 * total_missing / total_cells, 4)),
        ("registros_com_ausente", records_with_missing),
        ("pct_registros_com_ausente", round(100 * records_with_missing / len(df), 4)),
        ("registros_completos", len(df) - records_with_missing),
        ("registros_duplicados", int(df.duplicated().sum())),
        ("registros_duplicados_nas_13_caracteristicas", int(df[FEATURE_COLUMNS].duplicated().sum())),
        ("registros_com_outlier_iqr_nas_quantitativas", records_with_outlier),
    ]
    for count, frequency in missing_per_record.value_counts().sort_index().items():
        rows.append((f"registros_com_{count}_ausentes", int(frequency)))
    for category, frequency in df["CHAS"].value_counts(dropna=False).sort_index().items():
        label = "ausente" if pd.isna(category) else str(int(category))
        rows.append((f"CHAS_{label}", int(frequency)))

    return pd.DataFrame(rows, columns=["metrica", "valor"])


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas por variável; CHAS recebe apenas medidas compatíveis com categorias."""
    rows = []
    for column in _analysis_columns(df):
        series = df[column]
        mode = series.mode()
        mode_value = float(mode.iloc[0]) if not mode.empty else np.nan
        row = {
            "variavel": column,
            "tipo": "categorica_binaria" if column in CATEGORICAL_FEATURES else "quantitativa",
            "no_modelo": column in FEATURE_COLUMNS,
            "dtype": str(series.dtype),
            "validos": int(series.notna().sum()),
            "ausentes": int(series.isna().sum()),
            "pct_ausentes": round(100 * series.isna().mean(), 4),
            "valores_distintos": int(series.nunique()),
            "moda": mode_value,
            "pct_moda": round(100 * (series == mode_value).sum() / series.notna().sum(), 4),
        }
        if column not in CATEGORICAL_FEATURES:
            row.update(
                {
                    "media": series.mean(),
                    "desvio_padrao": series.std(),
                    "minimo": series.min(),
                    "q1": series.quantile(0.25),
                    "mediana": series.median(),
                    "q3": series.quantile(0.75),
                    "maximo": series.max(),
                    "assimetria": series.skew(),
                    "curtose": series.kurt(),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Conta possíveis outliers pelo critério de 1,5 × IQR; nada é removido da base.

    CHAS fica de fora porque o critério não se aplica a um código categórico.
    """
    rows = []
    for column in _quantitative_columns(df):
        series = df[column].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = _iqr_limits(series)
        below = int((series < lower).sum())
        above = int((series > upper).sum())
        rows.append(
            {
                "variavel": column,
                "no_modelo": column in FEATURE_COLUMNS,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "limite_inferior": lower,
                "limite_superior": upper,
                "abaixo_do_limite": below,
                "acima_do_limite": above,
                "total_outliers": below + above,
                "pct_outliers": round(100 * (below + above) / len(series), 4),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pearson e Spearman entre variáveis quantitativas, sem CHAS.

    Usa pares completos. Spearman complementa Pearson por ser menos sensível à
    assimetria e por respeitar apenas a ordem de RAD.
    """
    columns = _quantitative_columns(df)
    return {
        "pearson": df[columns].corr(method="pearson"),
        "spearman": df[columns].corr(method="spearman"),
    }


def plot_missing_values(df: pd.DataFrame, path: Path) -> None:
    """Gráfico de barras com os valores ausentes por coluna."""
    columns = _analysis_columns(df)
    counts = df[columns].isna().sum()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(columns[::-1], counts[columns[::-1]], color=COLORS["primary"], height=0.6)
    for bar, column in zip(bars, columns[::-1]):
        count = int(counts[column])
        pct = 100 * count / len(df)
        ax.annotate(
            f"{count} ({pct:.2f}%)",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=COLORS["ink_secondary"],
        )
    ax.set_xlim(0, max(1, counts.max()) * 1.2)
    ax.set_xlabel("Valores ausentes")
    ax.set_title(f"Valores ausentes por coluna ({len(df)} registros)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, path: Path) -> None:
    """Histogramas das quantitativas, barras para RAD (discreta) e para CHAS (categórica)."""
    columns = _analysis_columns(df)
    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.8 * n_rows))
    axes = axes.ravel()
    for ax, column in zip(axes, columns):
        series = df[column].dropna()
        if column in CATEGORICAL_FEATURES or column == "RAD":
            frequencies = series.astype(int).value_counts().sort_index()
            ax.bar(
                frequencies.index.astype(str),
                frequencies.to_numpy(),
                color=COLORS["primary"],
                width=0.7,
            )
            ax.grid(axis="x", visible=False)
            kind = "frequência" if column == "RAD" else "categórica, frequência"
            title = f"{column} ({kind})"
        else:
            ax.hist(series, bins=30, color=COLORS["primary"], edgecolor=COLORS["surface"], linewidth=0.5)
            title = f"{column} (assimetria {series.skew():.2f})"
            ax.grid(axis="x", visible=False)
        if column == TARGET_COLUMN:
            title = f"{column} (fora do modelo)"
        ax.set_title(title)
        ax.tick_params(labelsize=8)
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    fig.suptitle("Distribuição das variáveis (valores ausentes desconsiderados)", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_boxplots(df: pd.DataFrame, path: Path) -> None:
    """Boxplots das variáveis quantitativas, cada uma em sua própria escala."""
    columns = _quantitative_columns(df)
    n_cols = 5
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.1 * n_rows))
    axes = axes.ravel()
    for ax, column in zip(axes, columns):
        ax.boxplot(
            df[column].dropna(),
            widths=0.5,
            medianprops={"color": COLORS["accent"], "linewidth": 1.8},
            boxprops={"color": COLORS["primary"]},
            whiskerprops={"color": COLORS["primary"]},
            capprops={"color": COLORS["primary"]},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": COLORS["muted"],
                "markeredgecolor": "none",
                "alpha": 0.6,
            },
        )
        ax.set_title(column if column != TARGET_COLUMN else f"{column} (fora do modelo)")
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=8)
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    fig.suptitle("Boxplots das variáveis quantitativas (critério de 1,5 × IQR)", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation_matrix(matrices: dict[str, pd.DataFrame], path: Path) -> None:
    """Matrizes de correlação lado a lado, com as variáveis utilizadas explícitas."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.2))
    image = None
    for ax, (method, matrix) in zip(axes, matrices.items()):
        image = ax.imshow(matrix.to_numpy(), cmap=CORRELATION_CMAP, vmin=-1, vmax=1)
        labels = list(matrix.columns)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.tick_params(colors=COLORS["ink_secondary"], length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(len(labels)):
            for col in range(len(labels)):
                value = matrix.iat[row, col]
                ink = "#ffffff" if abs(value) >= 0.65 else COLORS["ink"]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color=ink)
        ax.set_title(method.capitalize())
    variables = ", ".join(next(iter(matrices.values())).columns)
    fig.suptitle(
        "Correlação entre variáveis quantitativas (pares completos)\n"
        f"Variáveis: {variables}. CHAS excluída por ser categórica; MEDV apenas descritiva.",
        fontweight="semibold",
    )
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda(df: pd.DataFrame, reports_dir: Path = REPORTS_DIR) -> dict[str, pd.DataFrame]:
    """Calcula as estatísticas, salva tabelas e gráficos e retorna as tabelas."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    summary = summarize_dataset(df)
    statistics = descriptive_statistics(df)
    outliers = outlier_summary(df)
    matrices = correlation_matrices(df)

    summary.to_csv(reports_dir / "eda_summary.csv", index=False)
    statistics.to_csv(reports_dir / "descriptive_statistics.csv", index=False)
    outliers.to_csv(reports_dir / "outlier_summary.csv", index=False)
    pd.concat(matrices, names=["metodo", "variavel"]).to_csv(reports_dir / "correlation_matrix.csv")

    plot_missing_values(df, reports_dir / "missing_values.png")
    plot_feature_distributions(df, reports_dir / "feature_distributions.png")
    plot_feature_boxplots(df, reports_dir / "feature_boxplots.png")
    plot_correlation_matrix(matrices, reports_dir / "correlation_matrix.png")

    return {"summary": summary, "statistics": statistics, "outliers": outliers}


def main(argv: list[str] | None = None) -> int:
    """Executa a análise exploratória isoladamente."""
    configure_console()
    parser = argparse.ArgumentParser(description="Análise exploratória da base Boston Housing.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Caminho do CSV.")
    args = parser.parse_args(argv)
    try:
        df = load_dataset(args.data)
    except (FileNotFoundError, DataValidationError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    tables = run_eda(df)
    print(tables["summary"].to_string(index=False))
    print(f"\nRelatórios salvos em: {REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
