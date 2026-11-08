"""Treinamento: análise de K, modelo definitivo, relatórios e persistência do artefato."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.pipeline import Pipeline

from src.config import (
    ALGORITHM,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    EXPERIMENTS_DIR,
    FEATURE_COLUMNS,
    FINAL_K,
    K_MAX,
    K_MIN,
    MAX_ITER,
    MIN_CLUSTER_PCT,
    MODEL_PATH,
    N_BOOTSTRAP,
    N_INIT,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    configure_console,
)
from src.eda import COLORS, apply_plot_style, run_eda
from src.preprocessing import (
    DataValidationError,
    build_preprocessor,
    denormalize_centroids,
    load_dataset,
    select_features,
)


def make_kmeans(k: int, random_state: int = RANDOM_STATE) -> KMeans:
    """Cria um KMeans com a configuração reproduzível do projeto."""
    return KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        algorithm=ALGORITHM,
    )


def build_model_pipeline(k: int) -> Pipeline:
    """Monta o pipeline completo salvo no artefato: pré-processador + KMeans."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("kmeans", make_kmeans(k))])


def bootstrap_stability(
    X: np.ndarray, reference_labels: np.ndarray, k: int, n_bootstrap: int
) -> tuple[float, float]:
    """Mede o quanto a partição se mantém quando a base é reamostrada.

    Cada reamostra treina um KMeans, que então rotula todos os registros; o ARI compara
    esses rótulos com a partição de referência. Retorna a média e o percentil 5.
    """
    rng = np.random.default_rng([RANDOM_STATE, k])
    scores = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(X), len(X))
        seed = int(rng.integers(0, 2**31 - 1))
        model = make_kmeans(k, random_state=seed).fit(X[indices])
        scores.append(adjusted_rand_score(reference_labels, model.predict(X)))
    return float(np.mean(scores)), float(np.percentile(scores, 5))


def run_k_analysis(
    X: np.ndarray,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """Avalia cada K: inércia, distância média, Silhouette (K >= 2), tamanhos e estabilidade.

    A inércia é a soma dos quadrados das distâncias euclidianas de cada registro ao seu
    centroide; a distância média é a média dessas distâncias, sem elevar ao quadrado. As
    duas curvas recebem o mesmo cálculo geométrico de cotovelo.
    """
    k_max = min(k_max, len(X) - 1)
    if k_max <= k_min:
        raise ValueError("A base não tem registros suficientes para avaliar a faixa de K.")

    rows: list[dict[str, Any]] = []
    for k in range(k_min, k_max + 1):
        model = make_kmeans(k).fit(X)
        sizes = np.bincount(model.labels_, minlength=k)
        point_distances = np.linalg.norm(X - model.cluster_centers_[model.labels_], axis=1)
        row: dict[str, Any] = {
            "k": k,
            "inercia": model.inertia_,
            "distancia_media_ao_centroide": point_distances.mean(),
            "silhouette": np.nan,
            "menor_cluster_n": int(sizes.min()),
            "menor_cluster_pct": 100 * sizes.min() / len(X),
            "tamanhos": " ".join(str(size) for size in sorted(sizes.tolist(), reverse=True)),
            "ari_bootstrap_medio": np.nan,
            "ari_bootstrap_p5": np.nan,
            "iteracoes": int(model.n_iter_),
        }
        if k >= 2:
            row["silhouette"] = silhouette_score(X, model.labels_)
            if n_bootstrap > 0:
                mean, p5 = bootstrap_stability(X, model.labels_, k, n_bootstrap)
                row["ari_bootstrap_medio"] = mean
                row["ari_bootstrap_p5"] = p5
        rows.append(row)

    results = pd.DataFrame(rows)
    total_inertia = results["inercia"].iloc[0]
    results.insert(2, "reducao_vs_k_anterior_pct", -100 * results["inercia"].pct_change())
    results.insert(3, "inercia_explicada_pct", 100 * (1 - results["inercia"] / total_inertia))
    _, distances = find_elbow(results["k"].to_numpy(), results["inercia"].to_numpy())
    results.insert(4, "distancia_cotovelo", distances)
    _, mean_distances = find_elbow(
        results["k"].to_numpy(), results["distancia_media_ao_centroide"].to_numpy()
    )
    results.insert(6, "distancia_cotovelo_dist_media", mean_distances)
    return results


def find_elbow(ks: np.ndarray, inertias: np.ndarray) -> tuple[int, np.ndarray]:
    """Identifica o cotovelo pela maior distância perpendicular à reta entre os extremos.

    K e inércia são normalizados para [0, 1], pois têm escalas incomparáveis. A distância
    de cada ponto à reta é |v × u|, em que v liga o primeiro ponto ao ponto avaliado e u é
    o vetor unitário da reta.
    """
    ks = np.asarray(ks, dtype=float)
    inertias = np.asarray(inertias, dtype=float)
    if len(ks) < 3:
        raise ValueError("São necessários ao menos três valores de K para localizar o cotovelo.")

    inertia_span = inertias.max() - inertias.min()
    if inertia_span == 0:
        return int(ks[0]), np.zeros(len(ks))

    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (inertias - inertias.min()) / inertia_span
    points = np.column_stack([x, y])
    line = points[-1] - points[0]
    unit = line / np.linalg.norm(line)
    relative = points - points[0]
    distances = np.abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
    return int(ks[int(np.argmax(distances))]), distances


def elbow_range_sensitivity(results: pd.DataFrame) -> dict[str, int]:
    """Recalcula o cotovelo em subfaixas, pois o método depende dos pontos extremos."""
    k_first = int(results["k"].min())
    k_last = int(results["k"].max())
    ranges = [(k_first, last) for last in range(max(k_first + 2, 6), k_last + 1)]
    ranges.append((k_first + 1, k_last))

    sensitivity = {}
    for low, high in ranges:
        subset = results[results["k"].between(low, high)]
        if len(subset) >= 3:
            elbow, _ = find_elbow(subset["k"].to_numpy(), subset["inercia"].to_numpy())
            sensitivity[f"{low}-{high}"] = elbow
    return sensitivity


def select_k(results: pd.DataFrame, k_override: int | None = None) -> dict[str, Any]:
    """Aplica a regra de decisão de K e registra as divergências entre os indicadores.

    Regra: adota o cotovelo geométrico se nenhum cluster ficar abaixo de MIN_CLUSTER_PCT;
    caso contrário, o maior Silhouette entre os K que respeitam esse mínimo. A
    interpretabilidade e a estabilidade pesam na decisão final, tomada pelo responsável
    em FINAL_K ou `--k`; o resultado da regra fica registrado mesmo quando substituído.
    """
    elbow_k, _ = find_elbow(results["k"].to_numpy(), results["inercia"].to_numpy())
    evaluated = results.dropna(subset=["silhouette"])
    eligible = evaluated[evaluated["menor_cluster_pct"] >= MIN_CLUSTER_PCT]
    best_silhouette_k = int(evaluated.loc[evaluated["silhouette"].idxmax(), "k"])

    rule_k, rule_criterion = None, None
    if elbow_k in set(eligible["k"]):
        rule_k = elbow_k
        rule_criterion = "cotovelo geométrico com tamanho mínimo de cluster atendido"
    elif not eligible.empty:
        rule_k = int(eligible.loc[eligible["silhouette"].idxmax(), "k"])
        rule_criterion = "maior Silhouette entre os K com tamanho mínimo de cluster atendido"

    if k_override is not None:
        if k_override not in set(evaluated["k"]):
            raise ValueError(f"K deve estar entre os valores avaliados: {sorted(evaluated['k'])}")
        selected = k_override
        # Um K fixado que coincide com a regra registra o critério da regra, e não uma escolha
        # arbitrária; qualquer outro K fica identificado como decisão do responsável.
        if k_override == rule_k:
            criterion = f"{rule_criterion} (confirmado em FINAL_K ou --k)"
        else:
            criterion = "decisão do responsável (FINAL_K ou --k)"
    elif rule_k is not None:
        selected, criterion = rule_k, rule_criterion
    else:
        raise ValueError(f"Nenhum K mantém todos os clusters acima de {MIN_CLUSTER_PCT}%.")

    mean_distance_elbow_k = None
    if "distancia_media_ao_centroide" in results:
        mean_distance_elbow_k, _ = find_elbow(
            results["k"].to_numpy(), results["distancia_media_ao_centroide"].to_numpy()
        )

    silhouette_rank = evaluated["silhouette"].rank(ascending=False, method="min")
    selected_row = evaluated[evaluated["k"] == selected].iloc[0]
    return {
        "k_selecionado": int(selected),
        "criterio": criterion,
        "k_regra_automatica": rule_k,
        "k_cotovelo_geometrico": elbow_k,
        "k_cotovelo_distancia_media": mean_distance_elbow_k,
        "k_maior_silhouette": best_silhouette_k,
        "indicadores_divergem": elbow_k != best_silhouette_k,
        "posicao_silhouette_do_k_selecionado": int(silhouette_rank[selected_row.name]),
        "tamanho_minimo_de_cluster_pct": MIN_CLUSTER_PCT,
        "cotovelo_por_faixa_de_k": elbow_range_sensitivity(results),
    }


def most_frequent_rad(features: pd.DataFrame, labels: np.ndarray) -> dict[int, dict[str, float]]:
    """RAD mais frequente nos registros reais de cada cluster (estatística complementar)."""
    summary = {}
    for cluster, values in features["RAD"].groupby(labels):
        frequencies = values.dropna().value_counts()
        summary[int(cluster)] = {
            "valor": float(frequencies.index[0]),
            "pct": float(100 * frequencies.iloc[0] / frequencies.sum()),
        }
    return summary


def distance_statistics(
    X: np.ndarray, labels: np.ndarray, centers: np.ndarray
) -> pd.DataFrame:
    """Distâncias euclidianas dos registros ao próprio centroide, por cluster."""
    distances = np.linalg.norm(X - centers[labels], axis=1)
    grouped = pd.Series(distances).groupby(labels)
    return pd.DataFrame(
        {
            "distancia_media": grouped.mean(),
            "distancia_p95": grouped.quantile(0.95),
            "distancia_maxima": grouped.max(),
        }
    ).rename_axis("cluster")


def cluster_distribution(labels: np.ndarray, k: int) -> pd.DataFrame:
    """Quantidade e percentual de registros por cluster."""
    counts = np.bincount(labels, minlength=k)
    return pd.DataFrame(
        {
            "cluster": range(k),
            "registros": counts,
            "percentual": 100 * counts / len(labels),
        }
    )


def centroid_table(
    profiles: list[dict[str, Any]], rad_modes: dict[int, dict[str, float]]
) -> pd.DataFrame:
    """Organiza os centroides na escala original para exportação."""
    rows = []
    for profile in profiles:
        cluster = profile["cluster"]
        chas = profile["CHAS"]
        rows.append(
            {
                "cluster": cluster,
                **profile["numeric"],
                "CHAS_0_proporcao": chas["proporcoes"][0],
                "CHAS_1_proporcao": chas["proporcoes"][1],
                "CHAS_predominante": chas["predominante"],
                "CHAS_predominante_pct": chas["predominante_pct"],
                "RAD_mais_frequente_nos_registros": rad_modes[cluster]["valor"],
            }
        )
    return pd.DataFrame(rows)


def cluster_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    df: pd.DataFrame,
    rad_modes: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """Reúne a qualidade interna de cada cluster e o perfil a posteriori de MEDV.

    MEDV só é consultada aqui, depois de o K e o modelo estarem definidos: ela descreve os
    grupos, sem influenciar sua formação.
    """
    table = distance_statistics(X, labels, centers).reset_index()
    silhouettes = pd.Series(silhouette_samples(X, labels)).groupby(labels)
    table.insert(1, "registros", np.bincount(labels, minlength=len(centers)))
    table["silhouette_medio"] = silhouettes.mean().to_numpy()
    table["silhouette_negativos"] = silhouettes.apply(lambda values: int((values < 0).sum())).to_numpy()
    table["RAD_mais_frequente"] = [rad_modes[cluster]["valor"] for cluster in table["cluster"]]
    table["RAD_mais_frequente_pct"] = [rad_modes[cluster]["pct"] for cluster in table["cluster"]]
    if TARGET_COLUMN in df.columns:
        target = df[TARGET_COLUMN].groupby(labels)
        table["MEDV_medio_a_posteriori"] = target.mean().to_numpy()
        table["MEDV_mediano_a_posteriori"] = target.median().to_numpy()
    return table


def feature_influence(
    pipeline: Pipeline, X: np.ndarray, labels: np.ndarray, features: pd.DataFrame
) -> pd.DataFrame:
    """Mede o peso de cada variável na separação e o efeito da padronização.

    `eta2_clusters` é a fração da variância da variável explicada pelos clusters, e
    `pct_soma_quadrados_entre` é sua parcela na separação total. `pct_variancia_original`
    mostra o domínio que a variável teria sem o StandardScaler.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_slice = preprocessor.output_indices_["numeric"]
    categorical_slice = preprocessor.output_indices_["categorical"]

    overall = X.mean(axis=0)
    total_ss = ((X - overall) ** 2).sum(axis=0)
    between_ss = np.zeros(X.shape[1])
    for cluster in np.unique(labels):
        members = X[labels == cluster]
        between_ss += len(members) * (members.mean(axis=0) - overall) ** 2

    raw_variance = features[NUMERIC_FEATURES].var()
    rows = []
    for name, index in zip(NUMERIC_FEATURES, range(*numeric_slice.indices(X.shape[1]))):
        rows.append(
            {
                "variavel": name,
                "variancia_original": raw_variance[name],
                "pct_variancia_original": 100 * raw_variance[name] / raw_variance.sum(),
                "eta2_clusters": between_ss[index] / total_ss[index],
                "pct_soma_quadrados_entre": 100 * between_ss[index] / between_ss.sum(),
            }
        )
    chas_between = between_ss[categorical_slice].sum()
    rows.append(
        {
            "variavel": CATEGORICAL_FEATURES[0],
            "variancia_original": np.nan,
            "pct_variancia_original": np.nan,
            "eta2_clusters": chas_between / total_ss[categorical_slice].sum(),
            "pct_soma_quadrados_entre": 100 * chas_between / between_ss.sum(),
        }
    )
    return pd.DataFrame(rows).sort_values("pct_soma_quadrados_entre", ascending=False)


def _plot_elbow_curve(
    ax: plt.Axes,
    results: pd.DataFrame,
    column: str,
    label: str,
    selected: int,
    elbow: int,
    value_format: str,
) -> None:
    """Desenha uma curva do cotovelo, a reta entre extremos, o K selecionado e o cotovelo."""
    point = results[results["k"] == selected].iloc[0]
    ends = results.iloc[[0, -1]]
    ax.plot(
        ends["k"],
        ends[column],
        linestyle="--",
        linewidth=1.3,
        color=COLORS["muted"],
        label="Reta entre os pontos extremos",
    )
    ax.plot(
        results["k"],
        results[column],
        marker="o",
        markersize=6,
        linewidth=2,
        color=COLORS["primary"],
        label=label,
    )
    ax.axvline(selected, linestyle=":", linewidth=1.2, color=COLORS["accent"])
    selected_label = f"K selecionado = {selected}"
    if elbow == selected:
        selected_label += " (cotovelo geométrico)"
    ax.scatter(
        [selected],
        [point[column]],
        s=130,
        color=COLORS["accent"],
        edgecolor=COLORS["surface"],
        linewidth=2,
        zorder=5,
        label=selected_label,
    )
    if elbow != selected:
        ax.scatter(
            [elbow],
            results.loc[results["k"] == elbow, column],
            s=110,
            facecolor=COLORS["surface"],
            edgecolor=COLORS["primary"],
            linewidth=2,
            zorder=5,
            label=f"Cotovelo geométrico = {elbow}",
        )
    ax.annotate(
        f"K = {selected}\n{label.lower()} = {point[column]:{value_format}}",
        (selected, point[column]),
        xytext=(12, 14),
        textcoords="offset points",
        color=COLORS["ink_secondary"],
    )
    ax.set_xticks(results["k"])
    ax.set_xlabel("Quantidade de clusters (K)")
    ax.set_ylabel(label)
    ax.legend(loc="upper right")


def plot_elbow(results: pd.DataFrame, selection: dict[str, Any], path: Path) -> None:
    """Curvas do cotovelo pela inércia e pela distância média ao centroide."""
    selected = selection["k_selecionado"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5.4))
    _plot_elbow_curve(
        left, results, "inercia", "Inércia", selected, selection["k_cotovelo_geometrico"], ".1f"
    )
    left.set_title("Inércia (soma dos quadrados das distâncias)")
    _plot_elbow_curve(
        right,
        results,
        "distancia_media_ao_centroide",
        "Distância média",
        selected,
        selection["k_cotovelo_distancia_media"],
        ".3f",
    )
    right.set_title("Distância média dos registros ao centroide")
    fig.suptitle("Método do cotovelo por quantidade de clusters", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_silhouette(results: pd.DataFrame, selection: dict[str, Any], path: Path) -> None:
    """Silhouette por K, com destaque para o K selecionado."""
    evaluated = results.dropna(subset=["silhouette"])
    selected = selection["k_selecionado"]
    point = evaluated[evaluated["k"] == selected].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(
        evaluated["k"],
        evaluated["silhouette"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=COLORS["primary"],
        label="Silhouette",
    )
    ax.scatter(
        [selected],
        [point["silhouette"]],
        s=130,
        color=COLORS["accent"],
        edgecolor=COLORS["surface"],
        linewidth=2,
        zorder=5,
        label=f"K selecionado = {selected}",
    )
    ax.set_xticks(evaluated["k"])
    ax.set_xlabel("Quantidade de clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score por quantidade de clusters")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cluster_distribution(distribution: pd.DataFrame, path: Path) -> None:
    """Barras com a quantidade de registros por cluster."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(
        distribution["cluster"].astype(str),
        distribution["registros"],
        color=COLORS["primary"],
        width=0.6,
    )
    for bar, row in zip(bars, distribution.itertuples()):
        ax.annotate(
            f"{row.registros} ({row.percentual:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color=COLORS["ink_secondary"],
        )
    ax.set_ylim(0, distribution["registros"].max() * 1.15)
    ax.set_xlabel("Cluster (identificador, não representa qualidade)")
    ax.set_ylabel("Registros")
    ax.set_title("Distribuição dos registros por cluster")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_metadata(
    pipeline: Pipeline,
    X: np.ndarray,
    features: pd.DataFrame,
    rad_modes: dict[int, dict[str, float]],
    selection: dict[str, Any],
    data_path: Path,
) -> dict[str, Any]:
    """Reúne o que a inferência precisa, sem nenhum registro individual da base."""
    preprocessor = pipeline.named_steps["preprocessor"]
    kmeans = pipeline.named_steps["kmeans"]
    numeric_slice = preprocessor.output_indices_["numeric"]
    categorical_slice = preprocessor.output_indices_["categorical"]
    distances = distance_statistics(X, kmeans.labels_, kmeans.cluster_centers_)
    return {
        "k": int(kmeans.n_clusters),
        "n_registros_treinamento": len(features),
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_columns": [TARGET_COLUMN],
        "centroid_layout": {
            "numeric_columns": [numeric_slice.start, numeric_slice.stop],
            "categorical_columns": [categorical_slice.start, categorical_slice.stop],
            "chas_categories": [int(value) for value in preprocessor.named_transformers_[
                "categorical"
            ].named_steps["encoder"].categories_[0]],
        },
        "training_config": {
            "random_state": RANDOM_STATE,
            "n_init": N_INIT,
            "max_iter": MAX_ITER,
            "algorithm": ALGORITHM,
        },
        "metrics": {
            "inercia": float(kmeans.inertia_),
            "silhouette": float(silhouette_score(X, kmeans.labels_)),
            "iteracoes": int(kmeans.n_iter_),
        },
        "selection": selection,
        "train_ranges": {
            column: {"min": float(features[column].min()), "max": float(features[column].max())}
            for column in NUMERIC_FEATURES
        },
        "rad_observed_values": sorted(float(value) for value in features["RAD"].dropna().unique()),
        "rad_mode_by_cluster": rad_modes,
        "distance_by_cluster": distances.to_dict(orient="index"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
    }


def save_artifact(pipeline: Pipeline, metadata: dict[str, Any], path: Path) -> None:
    """Salva o pipeline completo e os metadados em um único arquivo Joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, path)


def train(
    data_path: Path = DATA_PATH,
    k_override: int | None = None,
    reports_dir: Path = REPORTS_DIR,
    model_path: Path = MODEL_PATH,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, Any]:
    """Executa todo o treinamento e retorna um resumo dos resultados."""
    reports_dir = Path(reports_dir)
    df = load_dataset(data_path)
    run_eda(df, reports_dir)
    apply_plot_style()

    features = select_features(df)
    X = build_preprocessor().fit_transform(features)

    results = run_k_analysis(X, n_bootstrap=n_bootstrap)
    selection = select_k(results, k_override)
    results.to_csv(reports_dir / "elbow_results.csv", index=False, float_format="%.6f")
    (reports_dir / "k_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    plot_elbow(results, selection, reports_dir / "elbow_method.png")
    plot_silhouette(results, selection, reports_dir / "silhouette_scores.png")

    k = selection["k_selecionado"]
    pipeline = build_model_pipeline(k).fit(features)
    kmeans = pipeline.named_steps["kmeans"]
    labels = kmeans.labels_

    distribution = cluster_distribution(labels, k)
    distribution.to_csv(reports_dir / "cluster_distribution.csv", index=False, float_format="%.4f")
    plot_cluster_distribution(distribution, reports_dir / "cluster_distribution.png")

    rad_modes = most_frequent_rad(features, labels)
    profiles = denormalize_centroids(pipeline)
    centroid_table(profiles, rad_modes).to_csv(
        reports_dir / "cluster_centroids.csv", index=False, float_format="%.6f"
    )
    cluster_analysis(X, labels, kmeans.cluster_centers_, df, rad_modes).to_csv(
        reports_dir / "cluster_analysis.csv", index=False, float_format="%.6f"
    )
    feature_influence(pipeline, X, labels, features).to_csv(
        reports_dir / "feature_influence.csv", index=False, float_format="%.6f"
    )

    metadata = build_metadata(pipeline, X, features, rad_modes, selection, data_path)
    save_artifact(pipeline, metadata, model_path)
    return {"metadata": metadata, "distribution": distribution, "results": results}


def print_summary(summary: dict[str, Any], model_path: Path) -> None:
    """Exibe o resumo do treinamento no terminal."""
    metadata = summary["metadata"]
    selection = metadata["selection"]
    print(f"Registros utilizados: {metadata['n_registros_treinamento']}")
    print(f"K selecionado: {metadata['k']} ({selection['criterio']})")
    print(f"K pela regra automática: {selection['k_regra_automatica']}")
    print(f"K pelo cotovelo geométrico (inércia): {selection['k_cotovelo_geometrico']}")
    print(f"K pelo cotovelo geométrico (distância média): {selection['k_cotovelo_distancia_media']}")
    print(f"K com maior Silhouette: {selection['k_maior_silhouette']}")
    if selection["indicadores_divergem"]:
        print("Atenção: cotovelo e Silhouette divergem. Consulte a justificativa no README.")
    print(f"Inércia: {metadata['metrics']['inercia']:.4f}")
    print(f"Silhouette: {metadata['metrics']['silhouette']:.4f}")
    print("\nRegistros por cluster:")
    print(summary["distribution"].to_string(index=False, float_format="%.2f"))
    print(f"\nModelo salvo em: {model_path}")


def parse_k(text: str) -> int:
    """Valida `--k` antes de qualquer processamento: inteiro entre 2 e K_MAX.

    K = 1 não tem Silhouette e K acima de K_MAX não é avaliado na análise de K.
    """
    try:
        k = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' não é um número inteiro.") from None
    if not 2 <= k <= K_MAX:
        raise argparse.ArgumentTypeError(f"K deve ser um inteiro entre 2 e {K_MAX}; recebido {k}.")
    return k


def output_paths(k: int | None) -> tuple[Path, Path]:
    """Diretório de relatórios e caminho do modelo para o K solicitado.

    O K oficial (FINAL_K, ou a regra automática quando FINAL_K é None) grava em reports/ e
    models/. Qualquer outro K é experimental e grava em experiments/k<K>/, para nunca
    sobrescrever os resultados oficiais.
    """
    if k is None or k == FINAL_K:
        return REPORTS_DIR, MODEL_PATH
    experiment_dir = EXPERIMENTS_DIR / f"k{k}"
    return experiment_dir / "reports", experiment_dir / "kmeans_model.joblib"


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada de `python -m src.train`."""
    configure_console()
    parser = argparse.ArgumentParser(description="Treina o modelo K-Means da base Boston Housing.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Caminho alternativo do CSV.")
    parser.add_argument(
        "--k",
        type=parse_k,
        default=FINAL_K,
        help=(
            f"Quantidade de clusters, entre 2 e {K_MAX} (padrão: {FINAL_K}). Um K diferente "
            f"do padrão é experimental e grava em {EXPERIMENTS_DIR.name}/k<K>/."
        ),
    )
    args = parser.parse_args(argv)
    reports_dir, model_path = output_paths(args.k)
    if (reports_dir, model_path) != (REPORTS_DIR, MODEL_PATH):
        print(f"Treinamento experimental com K = {args.k}: os resultados oficiais não serão alterados.")
        print(f"Saída: {reports_dir.parent}\n")
    try:
        summary = train(
            data_path=args.data,
            k_override=args.k,
            reports_dir=reports_dir,
            model_path=model_path,
        )
    except (FileNotFoundError, DataValidationError, ValueError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    print_summary(summary, model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
