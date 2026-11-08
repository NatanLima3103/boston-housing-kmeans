"""Testes do pipeline.

Os componentes são exercitados com dados sintéticos, independentes da base real. Os
testes marcados com `needs_real_data` usam o CSV local e são ignorados se ele não existir,
já que o arquivo não é versionado.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from src import inference
from src.config import (
    CHAS_CATEGORIES,
    DATA_PATH,
    EXPERIMENTS_DIR,
    FEATURE_COLUMNS,
    FINAL_K,
    K_MAX,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.eda import descriptive_statistics, outlier_summary, run_eda, summarize_dataset
from src.inference import (
    InvalidObservationError,
    ModelLoadError,
    Observation,
    collect_interactive,
    format_result,
    load_artifact,
    parse_interactive_value,
    predict_observation,
    read_json_observation,
    validate_observation,
)
from src.preprocessing import (
    DataValidationError,
    build_preprocessor,
    denormalize_centroids,
    imputation_values,
    load_dataset,
    select_features,
)
from src.train import (
    bootstrap_stability,
    build_model_pipeline,
    elbow_range_sensitivity,
    find_elbow,
    main as train_main,
    make_kmeans,
    run_k_analysis,
    select_k,
    train,
)

needs_real_data = pytest.mark.skipif(
    not DATA_PATH.is_file(), reason="Base real ausente em data/HousingData.csv"
)

GROUP_SIZE = 30
K_SYNTHETIC = 3


def make_synthetic_frame() -> pd.DataFrame:
    """Cria três grupos bem separados, com valores ausentes e CHAS de composição conhecida."""
    rng = np.random.default_rng(7)
    centers = [
        {"CRIM": 0.1, "ZN": 60, "INDUS": 3, "NOX": 0.40, "RM": 7.0, "AGE": 20, "DIS": 8,
         "RAD": 2, "TAX": 250, "PTRATIO": 15, "B": 390, "LSTAT": 4},
        {"CRIM": 1.0, "ZN": 10, "INDUS": 10, "NOX": 0.55, "RM": 6.0, "AGE": 60, "DIS": 4,
         "RAD": 5, "TAX": 350, "PTRATIO": 18, "B": 350, "LSTAT": 12},
        {"CRIM": 15.0, "ZN": 0, "INDUS": 20, "NOX": 0.75, "RM": 5.0, "AGE": 95, "DIS": 1.5,
         "RAD": 24, "TAX": 660, "PTRATIO": 21, "B": 150, "LSTAT": 25},
    ]
    chas_by_group = [
        np.zeros(GROUP_SIZE),
        np.ones(GROUP_SIZE),
        np.array([1.0] * 6 + [0.0] * (GROUP_SIZE - 6)),
    ]
    frames = []
    for center, chas in zip(centers, chas_by_group):
        group = pd.DataFrame(
            {
                name: np.abs(value * (1 + 0.04 * rng.standard_normal(GROUP_SIZE)))
                for name, value in center.items()
            }
        )
        group["RAD"] = center["RAD"]
        group["CHAS"] = chas
        frames.append(group)

    df = pd.concat(frames, ignore_index=True)
    df[TARGET_COLUMN] = rng.uniform(5, 50, len(df))
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]
    df.loc[[1, 35, 70], "CRIM"] = np.nan
    df.loc[[2, 40], "LSTAT"] = np.nan
    df.loc[[3, 75], "CHAS"] = np.nan
    return df


def valid_observation() -> dict:
    """Observação sintética válida, próxima do primeiro grupo."""
    return {
        "identificacao": "Teste",
        "CRIM": 0.1, "ZN": 60.0, "INDUS": 3.0, "CHAS": 0, "NOX": 0.40, "RM": 7.0, "AGE": 20.0,
        "DIS": 8.0, "RAD": 2, "TAX": 250, "PTRATIO": 15.0, "B": 390.0, "LSTAT": 4.0,
    }


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_frame()


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory, synthetic_df):
    path = tmp_path_factory.mktemp("data") / "sintetico.csv"
    synthetic_df.to_csv(path, index=False, na_rep="NA")
    return path


@pytest.fixture(scope="module")
def trained(tmp_path_factory, synthetic_csv):
    """Treina uma vez com dados sintéticos, em diretórios temporários."""
    base = tmp_path_factory.mktemp("treino")
    model_path = base / "modelo.joblib"
    reports_dir = base / "reports"
    summary = train(
        data_path=synthetic_csv,
        k_override=K_SYNTHETIC,
        reports_dir=reports_dir,
        model_path=model_path,
        n_bootstrap=2,
    )
    return {"summary": summary, "model_path": model_path, "reports_dir": reports_dir}


@pytest.fixture(scope="module")
def artifact(trained):
    return load_artifact(trained["model_path"])


# 1. Leitura da base -------------------------------------------------------------------


def test_load_dataset_preserves_records_and_reads_na_as_missing(synthetic_csv, synthetic_df):
    df = load_dataset(synthetic_csv)
    assert df.shape == synthetic_df.shape
    assert df["CRIM"].isna().sum() == 3
    assert df["CHAS"].isna().sum() == 2


def test_load_dataset_missing_file_has_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="--data"):
        load_dataset(tmp_path / "inexistente.csv")


def test_load_dataset_rejects_empty_file(tmp_path):
    path = tmp_path / "vazio.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_dataset(path)


@needs_real_data
def test_real_dataset_matches_expected_version():
    df = load_dataset(DATA_PATH)
    summary = summarize_dataset(df).set_index("metrica")["valor"]
    assert df.shape == (506, 14)
    assert summary["celulas_ausentes"] == 120
    assert summary["registros_com_ausente"] == 112
    assert summary["registros_completos"] == 394
    assert summary["registros_duplicados"] == 0
    assert df.isna().sum().loc[["CRIM", "ZN", "INDUS", "CHAS", "AGE", "LSTAT"]].eq(20).all()


# 2. Validação das colunas -------------------------------------------------------------


@pytest.mark.parametrize("column", FEATURE_COLUMNS)
def test_missing_required_column_is_rejected(tmp_path, synthetic_df, column):
    path = tmp_path / "base.csv"
    synthetic_df.drop(columns=column).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match=column):
        load_dataset(path)


def test_medv_is_not_required(tmp_path, synthetic_df):
    path = tmp_path / "base.csv"
    synthetic_df.drop(columns=TARGET_COLUMN).to_csv(path, index=False)
    assert TARGET_COLUMN not in load_dataset(path).columns


def test_non_numeric_column_is_rejected(tmp_path, synthetic_df):
    path = tmp_path / "base.csv"
    broken = synthetic_df.astype({"RM": object})
    broken.loc[0, "RM"] = "texto"
    broken.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="RM"):
        load_dataset(path)


def test_invalid_chas_in_dataset_is_rejected(tmp_path, synthetic_df):
    path = tmp_path / "base.csv"
    broken = synthetic_df.copy()
    broken.loc[0, "CHAS"] = 2
    broken.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="CHAS"):
        load_dataset(path)


# 3. Exclusão de MEDV ------------------------------------------------------------------


def test_select_features_excludes_medv_and_keeps_order(synthetic_df):
    features = select_features(synthetic_df)
    assert TARGET_COLUMN not in features.columns
    assert list(features.columns) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 13


def test_medv_does_not_influence_model(synthetic_df):
    altered = synthetic_df.copy()
    altered[TARGET_COLUMN] = altered[TARGET_COLUMN].to_numpy()[::-1] * 100

    original_model = build_model_pipeline(K_SYNTHETIC).fit(select_features(synthetic_df))
    altered_model = build_model_pipeline(K_SYNTHETIC).fit(select_features(altered))

    np.testing.assert_array_equal(
        original_model.named_steps["kmeans"].cluster_centers_,
        altered_model.named_steps["kmeans"].cluster_centers_,
    )


def test_artifact_preprocessor_never_saw_medv(artifact):
    preprocessor = artifact["pipeline"].named_steps["preprocessor"]
    assert TARGET_COLUMN not in preprocessor.feature_names_in_
    assert artifact["metadata"]["excluded_columns"] == [TARGET_COLUMN]


# 4. Valores ausentes ------------------------------------------------------------------


def test_transform_has_no_missing_and_preserves_records(synthetic_df):
    features = select_features(synthetic_df)
    transformed = build_preprocessor().fit_transform(features)
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(synthetic_df)


def test_imputers_use_median_and_mode(synthetic_df):
    features = select_features(synthetic_df)
    pipeline = build_model_pipeline(K_SYNTHETIC).fit(features)
    imputed = imputation_values(pipeline)
    assert imputed["CRIM"] == pytest.approx(features["CRIM"].median())
    assert imputed["LSTAT"] == pytest.approx(features["LSTAT"].median())
    assert imputed["CHAS"] == features["CHAS"].mode().iloc[0]


# 5. Ajuste do pré-processador ---------------------------------------------------------


def test_preprocessor_scaler_learns_imputed_statistics(synthetic_df):
    features = select_features(synthetic_df)
    preprocessor = build_preprocessor().fit(features)
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    filled = features[NUMERIC_FEATURES].fillna(features[NUMERIC_FEATURES].median())
    np.testing.assert_allclose(scaler.mean_, filled.mean().to_numpy())
    np.testing.assert_allclose(scaler.scale_, filled.std(ddof=0).to_numpy())


def test_encoder_keeps_explicit_categories(synthetic_df):
    preprocessor = build_preprocessor().fit(select_features(synthetic_df))
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    assert encoder.categories_[0].tolist() == CHAS_CATEGORIES
    assert encoder.handle_unknown == "ignore"


# 6. Transformação ---------------------------------------------------------------------


def test_transform_layout_scales_numeric_and_keeps_rad_as_single_column(synthetic_df):
    transformed = build_preprocessor().fit_transform(select_features(synthetic_df))
    assert transformed.shape[1] == len(NUMERIC_FEATURES) + len(CHAS_CATEGORIES)
    numeric = transformed[:, : len(NUMERIC_FEATURES)]
    np.testing.assert_allclose(numeric.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(numeric.std(axis=0), 1)
    one_hot = transformed[:, len(NUMERIC_FEATURES):]
    assert set(np.unique(one_hot)) == {0.0, 1.0}
    np.testing.assert_allclose(one_hot.sum(axis=1), 1)


def test_chas_is_one_hot_encoded_without_scaling(synthetic_df):
    preprocessor = build_preprocessor().fit(select_features(synthetic_df))
    frame = pd.DataFrame([valid_observation()])[FEATURE_COLUMNS]
    transformed = preprocessor.transform(frame.astype(float))
    assert transformed[0, len(NUMERIC_FEATURES):].tolist() == [1.0, 0.0]


# 7. Método do cotovelo ----------------------------------------------------------------


def test_find_elbow_matches_manual_geometry():
    elbow, distances = find_elbow(np.array([1, 2, 3]), np.array([10.0, 0.0, 0.0]))
    assert elbow == 2
    assert distances[1] == pytest.approx(math.sqrt(2) / 4)
    assert distances[0] == pytest.approx(0)
    assert distances[2] == pytest.approx(0)


def test_find_elbow_is_scale_invariant():
    ks = np.arange(1, 7)
    inertias = np.array([100.0, 20, 15, 12, 10, 9])
    assert find_elbow(ks, inertias)[0] == find_elbow(ks, inertias * 1e6)[0] == 2


def test_find_elbow_handles_flat_curve():
    elbow, distances = find_elbow(np.array([1, 2, 3]), np.array([5.0, 5.0, 5.0]))
    assert elbow == 1
    assert not distances.any()


def test_run_k_analysis_reports_all_k(synthetic_df):
    X = build_preprocessor().fit_transform(select_features(synthetic_df))
    results = run_k_analysis(X, k_min=1, k_max=6, n_bootstrap=2)
    assert results["k"].tolist() == [1, 2, 3, 4, 5, 6]
    assert results["inercia"].is_monotonic_decreasing
    assert math.isnan(results.loc[0, "silhouette"])
    assert results.loc[1:, "silhouette"].notna().all()
    assert results.loc[0, "inercia_explicada_pct"] == pytest.approx(0)
    assert results.loc[results["silhouette"].idxmax(), "k"] == K_SYNTHETIC


def _analysis_table(smallest_pct: dict[int, float]) -> pd.DataFrame:
    """Tabela mínima com cotovelo em K=3 e maior Silhouette em K=2."""
    return pd.DataFrame(
        {
            "k": [1, 2, 3, 4, 5],
            "inercia": [100.0, 40.0, 15.0, 12.0, 10.0],
            "silhouette": [np.nan, 0.6, 0.4, 0.5, 0.3],
            "menor_cluster_pct": [100.0] + [smallest_pct[k] for k in (2, 3, 4, 5)],
        }
    )


def test_select_k_prefers_elbow_and_flags_divergence():
    selection = select_k(_analysis_table({2: 40, 3: 20, 4: 10, 5: 8}))
    assert selection["k_selecionado"] == 3
    assert selection["k_maior_silhouette"] == 2
    assert selection["indicadores_divergem"] is True
    assert selection["posicao_silhouette_do_k_selecionado"] == 3


def test_select_k_falls_back_when_elbow_has_tiny_cluster():
    selection = select_k(_analysis_table({2: 40, 3: 1, 4: 10, 5: 8}))
    assert selection["k_selecionado"] == 2
    assert "Silhouette" in selection["criterio"]


def test_select_k_override_is_recorded_and_validated():
    table = _analysis_table({2: 40, 3: 20, 4: 10, 5: 8})
    selection = select_k(table, k_override=2)
    assert selection["k_selecionado"] == 2
    assert selection["criterio"] == "decisão do responsável (FINAL_K ou --k)"
    assert selection["k_regra_automatica"] == 3
    with pytest.raises(ValueError):
        select_k(table, k_override=1)


def test_elbow_sensitivity_covers_alternative_ranges():
    table = _analysis_table({2: 40, 3: 20, 4: 10, 5: 8})
    assert elbow_range_sensitivity(table) == {"2-5": 3}


# 8. Treinamento do KMeans -------------------------------------------------------------


def test_kmeans_configuration_and_reproducibility(synthetic_df):
    features = select_features(synthetic_df)
    first = build_model_pipeline(K_SYNTHETIC).fit(features)
    second = build_model_pipeline(K_SYNTHETIC).fit(features)
    kmeans = first.named_steps["kmeans"]

    assert kmeans.random_state == 42
    assert kmeans.n_init == 20
    assert kmeans.max_iter == 300
    assert kmeans.algorithm == "lloyd"
    assert kmeans.cluster_centers_.shape == (K_SYNTHETIC, 14)
    np.testing.assert_array_equal(kmeans.labels_, second.named_steps["kmeans"].labels_)


def test_kmeans_recovers_synthetic_groups(synthetic_df):
    labels = build_model_pipeline(K_SYNTHETIC).fit(select_features(synthetic_df)).named_steps[
        "kmeans"
    ].labels_
    assert sorted(np.bincount(labels).tolist()) == [GROUP_SIZE] * K_SYNTHETIC
    for start in range(0, len(labels), GROUP_SIZE):
        assert len(set(labels[start : start + GROUP_SIZE])) == 1


def test_inertia_is_sum_of_squared_distances(synthetic_df):
    features = select_features(synthetic_df)
    pipeline = build_model_pipeline(K_SYNTHETIC).fit(features)
    kmeans = pipeline.named_steps["kmeans"]
    X = pipeline.named_steps["preprocessor"].transform(features)
    expected = ((X - kmeans.cluster_centers_[kmeans.labels_]) ** 2).sum()
    assert kmeans.inertia_ == pytest.approx(expected)


def test_bootstrap_stability_is_high_for_separated_groups(synthetic_df):
    X = build_preprocessor().fit_transform(select_features(synthetic_df))
    labels = make_kmeans(K_SYNTHETIC).fit(X).labels_
    mean, p5 = bootstrap_stability(X, labels, K_SYNTHETIC, n_bootstrap=5)
    assert mean == pytest.approx(1.0)
    assert p5 == pytest.approx(1.0)


def test_training_writes_expected_reports(trained):
    reports = trained["reports_dir"]
    expected = [
        "elbow_method.png", "elbow_results.csv", "silhouette_scores.png", "k_selection.json",
        "cluster_distribution.png", "cluster_distribution.csv", "cluster_centroids.csv",
        "cluster_analysis.csv", "feature_influence.csv", "eda_summary.csv",
        "descriptive_statistics.csv", "missing_values.png", "feature_distributions.png",
        "feature_boxplots.png", "correlation_matrix.png", "correlation_matrix.csv",
        "outlier_summary.csv",
    ]
    assert [name for name in expected if not (reports / name).is_file()] == []

    centroids = pd.read_csv(reports / "cluster_centroids.csv")
    assert TARGET_COLUMN not in centroids.columns
    assert len(centroids) == K_SYNTHETIC

    distribution = pd.read_csv(reports / "cluster_distribution.csv")
    assert distribution["registros"].sum() == 3 * GROUP_SIZE
    assert distribution["percentual"].sum() == pytest.approx(100, abs=0.01)


def test_k_selection_json_uses_lf_line_endings(trained):
    content = (trained["reports_dir"] / "k_selection.json").read_bytes()
    assert b"\r\n" not in content and b"\n" in content
    assert json.loads(content.decode("utf-8"))["k_selecionado"] == K_SYNTHETIC


def test_eda_does_not_treat_chas_as_continuous(tmp_path, synthetic_df):
    tables = run_eda(synthetic_df, tmp_path)
    assert "CHAS" not in set(tables["outliers"]["variavel"])
    statistics = tables["statistics"].set_index("variavel")
    assert statistics.loc["CHAS", "tipo"] == "categorica_binaria"
    assert math.isnan(statistics.loc["CHAS", "media"])
    correlation = pd.read_csv(tmp_path / "correlation_matrix.csv")
    assert "CHAS" not in correlation.columns


def test_outlier_summary_follows_iqr_rule():
    df = make_synthetic_frame()
    df["TAX"] = 100.0
    df.loc[0, "TAX"] = 10_000.0
    row = outlier_summary(df).set_index("variavel").loc["TAX"]
    assert row["acima_do_limite"] == 1
    assert row["abaixo_do_limite"] == 0


def test_descriptive_statistics_counts_missing(synthetic_df):
    statistics = descriptive_statistics(synthetic_df).set_index("variavel")
    assert statistics.loc["CRIM", "ausentes"] == 3
    assert statistics.loc["ZN", "ausentes"] == 0


# 9 e 10. Salvamento e carregamento ----------------------------------------------------


def test_artifact_contains_full_pipeline_and_metadata(trained):
    raw = joblib.load(trained["model_path"])
    pipeline = raw["pipeline"]
    metadata = raw["metadata"]

    assert list(pipeline.named_steps) == ["preprocessor", "kmeans"]
    preprocessor = pipeline.named_steps["preprocessor"]
    assert list(preprocessor.named_transformers_["numeric"].named_steps) == ["imputer", "scaler"]
    assert list(preprocessor.named_transformers_["categorical"].named_steps) == [
        "imputer",
        "encoder",
    ]
    assert metadata["k"] == K_SYNTHETIC
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert metadata["numeric_features"] == NUMERIC_FEATURES
    assert metadata["categorical_features"] == ["CHAS"]
    assert metadata["training_config"]["random_state"] == 42
    assert metadata["python_version"]
    assert metadata["sklearn_version"]
    assert metadata["centroid_layout"]["numeric_columns"] == [0, 12]
    assert metadata["centroid_layout"]["categorical_columns"] == [12, 14]


def test_load_artifact_missing_file_points_to_training(tmp_path):
    with pytest.raises(ModelLoadError, match="python -m src.train"):
        load_artifact(tmp_path / "ausente.joblib")


def test_load_artifact_rejects_corrupted_file(tmp_path):
    path = tmp_path / "corrompido.joblib"
    path.write_bytes(b"isto nao e um modelo")
    with pytest.raises(ModelLoadError) as raised:
        load_artifact(path)
    message = str(raised.value)
    assert "corrompido" in message and "python -m src.train" in message
    # Regressão: o detalhe interno do pickle fica só na exceção encadeada.
    assert raised.value.__cause__ is not None
    assert str(raised.value.__cause__) not in message


def test_load_artifact_rejects_unexpected_structure(tmp_path):
    path = tmp_path / "estrutura.joblib"
    joblib.dump({"modelo": 1}, path)
    with pytest.raises(ModelLoadError, match="artefato válido"):
        load_artifact(path)


def test_load_artifact_rejects_pipeline_of_wrong_type(tmp_path):
    path = tmp_path / "estrutura.joblib"
    joblib.dump({"pipeline": 1, "metadata": {}}, path)
    with pytest.raises(ModelLoadError, match="etapas esperadas"):
        load_artifact(path)


# 11. Inferência de uma nova observação ------------------------------------------------


def test_predict_observation_returns_complete_result(artifact):
    result = predict_observation(validate_observation(valid_observation()), artifact)
    assert result["identificacao"] == "Teste"
    assert 0 <= result["cluster"] < K_SYNTHETIC
    assert result["distancia"] >= 0
    assert set(result["centroide"]["numeric"]) == set(NUMERIC_FEATURES)
    assert result["avisos"] == []


def test_format_result_shows_required_sections(artifact):
    text = format_result(predict_observation(validate_observation(valid_observation()), artifact))
    for expected in (
        "RESULTADO DA INFERÊNCIA",
        "Identificação: Teste",
        "Cluster atribuído:",
        "Distância até o centroide:",
        "Espaço normalizado e codificado",
        "CENTROIDE NA ESCALA ORIGINAL",
        "Categoria 0:",
        "Categoria 1:",
        "Categoria predominante:",
    ):
        assert expected in text
    assert TARGET_COLUMN not in text


def test_inference_does_not_refit_preprocessor(artifact):
    preprocessor = artifact["pipeline"].named_steps["preprocessor"]
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    mean_before = scaler.mean_.copy()
    centers_before = artifact["pipeline"].named_steps["kmeans"].cluster_centers_.copy()

    extreme = valid_observation() | {"CRIM": 500.0, "TAX": 5000.0}
    predict_observation(validate_observation(extreme), artifact)

    np.testing.assert_array_equal(scaler.mean_, mean_before)
    np.testing.assert_array_equal(
        artifact["pipeline"].named_steps["kmeans"].cluster_centers_, centers_before
    )


def test_null_is_accepted_and_imputed_with_notice(artifact):
    raw = valid_observation() | {"CRIM": None, "CHAS": None}
    result = predict_observation(validate_observation(raw), artifact)
    messages = " ".join(result["avisos"])
    assert "CRIM ausente (null): imputado pela mediana" in messages
    assert "CHAS ausente (null): imputado pela moda" in messages


def test_null_imputation_equals_explicit_training_median(artifact):
    median = imputation_values(artifact["pipeline"])["CRIM"]
    with_null = predict_observation(
        validate_observation(valid_observation() | {"CRIM": None}), artifact
    )
    explicit = predict_observation(
        validate_observation(valid_observation() | {"CRIM": median}), artifact
    )
    assert with_null["cluster"] == explicit["cluster"]
    assert with_null["distancia"] == pytest.approx(explicit["distancia"])


def test_out_of_range_value_warns_without_rejecting(artifact):
    raw = valid_observation() | {"CRIM": 500.0, "RAD": 9}
    result = predict_observation(validate_observation(raw), artifact)
    messages = " ".join(result["avisos"])
    assert "CRIM = 500 está fora do intervalo" in messages
    assert "RAD = 9 não foi observado" in messages


def test_version_mismatch_produces_warning(artifact):
    changed = {
        "pipeline": artifact["pipeline"],
        "metadata": artifact["metadata"] | {"sklearn_version": "0.0.1"},
    }
    result = predict_observation(validate_observation(valid_observation()), changed)
    assert any("scikit-learn 0.0.1" in message for message in result["avisos"])


# 12. Entradas inválidas ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"MEDV": 20.0}, "MEDV não faz parte"),
        ({"EXTRA": 1}, "inesperadas"),
        ({"CRIM": "abc"}, "CRIM: esperado um número"),
        ({"CRIM": "0.05"}, "CRIM: esperado um número"),
        ({"RM": True}, "RM: esperado um número"),
        ({"RM": [1, 2]}, "RM: esperado um número"),
        ({"TAX": float("inf")}, "TAX: o valor deve ser finito"),
        ({"TAX": float("nan")}, "TAX: o valor deve ser finito"),
        ({"TAX": 10**400}, "TAX: número grande demais"),
        ({"CHAS": 2}, "CHAS: deve ser 0 ou 1"),
        ({"CHAS": 0.5}, "CHAS: deve ser 0 ou 1"),
        ({"CHAS": "1"}, "CHAS: esperado um número"),
        ({"CRIM": -1}, "CRIM: não pode ser negativo"),
        ({"AGE": 101}, "AGE: é um percentual"),
        ({"identificacao": 123}, "identificacao: deve ser um texto"),
    ],
)
def test_invalid_values_are_rejected(changes, message):
    with pytest.raises(InvalidObservationError, match=message):
        validate_observation(valid_observation() | changes)


def test_absent_field_differs_from_null():
    absent = valid_observation()
    del absent["LSTAT"]
    with pytest.raises(InvalidObservationError, match="ausentes.*LSTAT"):
        validate_observation(absent)

    observation = validate_observation(valid_observation() | {"LSTAT": None})
    assert math.isnan(observation.values["LSTAT"])


def test_all_problems_are_reported_together():
    raw = valid_observation() | {"CRIM": "abc", "CHAS": 2, "EXTRA": 1}
    with pytest.raises(InvalidObservationError) as error:
        validate_observation(raw)
    message = str(error.value)
    assert "CRIM" in message and "CHAS" in message and "EXTRA" in message


def test_non_object_input_is_rejected():
    with pytest.raises(InvalidObservationError, match="objeto JSON"):
        validate_observation([1, 2, 3])


def test_identification_is_optional():
    raw = valid_observation()
    del raw["identificacao"]
    assert validate_observation(raw).identificacao == inference.DEFAULT_ID


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{invalido", "JSON inválido"),
        ('{"CRIM": NaN}', "NaN"),
        ('{"CRIM": Infinity}', "Infinity"),
        ('{"CRIM": 1, "CRIM": 2}', "duplicados"),
    ],
)
def test_malformed_json_is_rejected(tmp_path, content, message):
    path = tmp_path / "entrada.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InvalidObservationError, match=message):
        read_json_observation(path)


def test_json_overflow_to_infinity_is_rejected(tmp_path):
    path = tmp_path / "entrada.json"
    path.write_text(json.dumps(valid_observation()).replace('"TAX": 250', '"TAX": 1e999'), "utf-8")
    with pytest.raises(InvalidObservationError, match="TAX"):
        validate_observation(read_json_observation(path))


def test_json_with_bom_is_accepted(tmp_path):
    path = tmp_path / "entrada.json"
    path.write_text(json.dumps(valid_observation()), encoding="utf-8-sig")
    assert read_json_observation(path)["CRIM"] == 0.1


def test_missing_json_file_is_reported(tmp_path):
    with pytest.raises(InvalidObservationError, match="Não foi possível ler"):
        read_json_observation(tmp_path / "ausente.json")


def test_interactive_collection_retries_and_never_asks_medv(capsys):
    answers = iter(
        ["  Regiao X ", "abc", "0,1", "60", "3", "2", "0", "0.4", "7", "20", "8", "2", "250", "15",
         "390", "null"]
    )
    prompts: list[str] = []

    def fake_prompt(text: str) -> str:
        prompts.append(text)
        return next(answers)

    observation = validate_observation(collect_interactive(fake_prompt))

    assert observation.identificacao == "Regiao X"
    assert observation.values["CRIM"] == pytest.approx(0.1)
    assert observation.values["CHAS"] == 0.0
    assert math.isnan(observation.values["LSTAT"])
    assert not any(prompt.startswith(TARGET_COLUMN) for prompt in prompts)
    assert "Valor inválido" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("text", "expected"), [("1,5", 1.5), (" 2.5 ", 2.5), ("NULL", None), ("na", None)]
)
def test_parse_interactive_value(text, expected):
    assert parse_interactive_value(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "abc", "1,2,3"])
def test_parse_interactive_value_rejects_invalid_text(text):
    with pytest.raises(InvalidObservationError):
        parse_interactive_value(text)


# 13. Identificação do cluster ---------------------------------------------------------


def test_assigned_cluster_is_nearest_centroid(artifact):
    pipeline = artifact["pipeline"]
    observation = validate_observation(valid_observation())
    result = predict_observation(observation, artifact)

    frame = pd.DataFrame([observation.values], columns=FEATURE_COLUMNS, dtype=float)
    transformed = pipeline.named_steps["preprocessor"].transform(frame)
    distances = np.linalg.norm(
        pipeline.named_steps["kmeans"].cluster_centers_ - transformed, axis=1
    )
    assert result["cluster"] == int(np.argmin(distances))
    assert result["distancia"] == pytest.approx(distances.min())


def test_observation_near_each_group_lands_in_its_cluster(artifact, synthetic_df):
    labels = artifact["pipeline"].named_steps["kmeans"].labels_
    for row in (0, GROUP_SIZE, 2 * GROUP_SIZE):
        values = select_features(synthetic_df).iloc[row].to_dict()
        result = predict_observation(Observation("x", values), artifact)
        assert result["cluster"] == labels[row]


# 14. Recuperação do centroide ---------------------------------------------------------


def test_returned_centroid_belongs_to_assigned_cluster(artifact):
    result = predict_observation(validate_observation(valid_observation()), artifact)
    profiles = denormalize_centroids(artifact["pipeline"])
    assert result["centroide"] == profiles[result["cluster"]]
    assert result["centroide"]["cluster"] == result["cluster"]


def test_distance_matches_kmeans_transform(artifact):
    observation = validate_observation(valid_observation())
    result = predict_observation(observation, artifact)
    frame = pd.DataFrame([observation.values], columns=FEATURE_COLUMNS, dtype=float)
    assert result["distancia"] == pytest.approx(
        artifact["pipeline"].transform(frame)[0, result["cluster"]]
    )


# 15. Desnormalização ------------------------------------------------------------------


def test_scaler_inverse_recovers_original_scale(synthetic_df):
    features = select_features(synthetic_df)
    preprocessor = build_preprocessor().fit(features)
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    transformed = preprocessor.transform(features)[:, preprocessor.output_indices_["numeric"]]
    filled = features[NUMERIC_FEATURES].fillna(features[NUMERIC_FEATURES].median())
    np.testing.assert_allclose(scaler.inverse_transform(transformed), filled.to_numpy())


def test_denormalized_centroids_equal_member_means(artifact, synthetic_df):
    pipeline = artifact["pipeline"]
    features = select_features(synthetic_df)
    labels = pipeline.named_steps["kmeans"].labels_
    filled = features[NUMERIC_FEATURES].fillna(features[NUMERIC_FEATURES].median())

    for profile in denormalize_centroids(pipeline):
        members = filled[labels == profile["cluster"]]
        for feature in NUMERIC_FEATURES:
            assert profile["numeric"][feature] == pytest.approx(members[feature].mean())


def test_all_zero_group_is_not_shown_as_negative_zero(artifact):
    values = [profile["numeric"]["ZN"] for profile in denormalize_centroids(artifact["pipeline"])]
    assert min(values) == 0
    assert "-" not in f"{min(values):.4f}"


def test_rad_centroid_is_not_rounded():
    df = make_synthetic_frame()
    df.loc[: GROUP_SIZE - 1, "RAD"] = np.tile([2, 3], GROUP_SIZE // 2)
    pipeline = build_model_pipeline(K_SYNTHETIC).fit(select_features(df))
    values = [profile["numeric"]["RAD"] for profile in denormalize_centroids(pipeline)]
    assert any(value != round(value) for value in values)


# 16. Proporções categóricas -----------------------------------------------------------


def test_chas_proportions_sum_to_one(artifact):
    for profile in denormalize_centroids(artifact["pipeline"]):
        assert sum(profile["CHAS"]["proporcoes"].values()) == pytest.approx(1)


def test_chas_proportions_match_imputed_members(artifact, synthetic_df):
    pipeline = artifact["pipeline"]
    labels = pipeline.named_steps["kmeans"].labels_
    chas = synthetic_df["CHAS"].fillna(imputation_values(pipeline)["CHAS"])

    for profile in denormalize_centroids(pipeline):
        members = chas[labels == profile["cluster"]]
        assert profile["CHAS"]["proporcoes"][1] == pytest.approx((members == 1).mean())
        assert profile["CHAS"]["proporcoes"][0] == pytest.approx((members == 0).mean())


def test_chas_dominant_category_is_consistent(artifact):
    for profile in denormalize_centroids(artifact["pipeline"]):
        chas = profile["CHAS"]
        assert chas["proporcoes"][chas["predominante"]] == max(chas["proporcoes"].values())
        assert chas["predominante_pct"] == pytest.approx(100 * max(chas["proporcoes"].values()))


# 17. Consistência entre treinamento e inferência --------------------------------------


def test_predict_on_training_records_matches_labels(artifact, synthetic_df):
    pipeline = artifact["pipeline"]
    np.testing.assert_array_equal(
        pipeline.predict(select_features(synthetic_df)), pipeline.named_steps["kmeans"].labels_
    )


def test_single_row_transform_matches_batch_transform(artifact, synthetic_df):
    preprocessor = artifact["pipeline"].named_steps["preprocessor"]
    features = select_features(synthetic_df)
    batch = preprocessor.transform(features)
    for row in (0, 1, 75):
        single = preprocessor.transform(features.iloc[[row]])
        np.testing.assert_allclose(single[0], batch[row])


def test_field_order_of_input_does_not_change_result(artifact):
    forward = valid_observation()
    backward = dict(reversed(list(forward.items())))
    first = predict_observation(validate_observation(forward), artifact)
    second = predict_observation(validate_observation(backward), artifact)
    assert first["cluster"] == second["cluster"]
    assert first["distancia"] == pytest.approx(second["distancia"])


def test_training_is_reproducible(tmp_path, synthetic_csv, trained):
    repeated = train(
        data_path=synthetic_csv,
        k_override=K_SYNTHETIC,
        reports_dir=tmp_path / "reports",
        model_path=tmp_path / "modelo.joblib",
        n_bootstrap=2,
    )
    original = trained["summary"]
    pd.testing.assert_frame_equal(repeated["results"], original["results"])
    assert repeated["metadata"]["metrics"] == original["metadata"]["metrics"]


def test_cli_loads_model_in_independent_process(trained, tmp_path):
    path = tmp_path / "entrada.json"
    path.write_text(json.dumps(valid_observation()), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "src.inference", "--json", str(path), "--model", str(trained["model_path"])],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert "Cluster atribuído:" in completed.stdout


def test_cli_reports_missing_model(tmp_path, capsys):
    code = inference.main(["--model", str(tmp_path / "ausente.joblib")])
    assert code == 1
    assert "python -m src.train" in capsys.readouterr().err


def test_cli_reports_invalid_input(trained, tmp_path, capsys):
    path = tmp_path / "entrada.json"
    path.write_text('{"CRIM": 1}', encoding="utf-8")
    code = inference.main(["--json", str(path), "--model", str(trained["model_path"])])
    assert code == 1
    assert "Campos obrigatórios ausentes" in capsys.readouterr().err


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Modelo ainda não treinado")
def test_saved_model_uses_final_k():
    metadata = load_artifact(MODEL_PATH)["metadata"]
    assert metadata["k"] == FINAL_K
    assert metadata["selection"]["k_selecionado"] == FINAL_K


def test_cli_default_k_is_final_k(monkeypatch, tmp_path):
    received = {}

    def fake_train(data_path, k_override, **paths):
        received["k"] = k_override
        received.update(paths)
        raise ValueError("interrompido pelo teste")

    monkeypatch.setattr("src.train.train", fake_train)
    assert train_main([]) == 1
    assert received["k"] == FINAL_K
    assert received["reports_dir"] == REPORTS_DIR
    assert received["model_path"] == MODEL_PATH

    received.clear()
    assert train_main(["--k", str(FINAL_K)]) == 1
    assert (received["reports_dir"], received["model_path"]) == (REPORTS_DIR, MODEL_PATH)


@pytest.mark.parametrize("value", ["1", "0", "-2", str(K_MAX + 1), "abc", "2.5"])
def test_cli_rejects_invalid_k_before_any_processing(monkeypatch, capsys, value):
    def fail_if_called(**kwargs):
        raise AssertionError("o treinamento não deveria começar com K inválido")

    monkeypatch.setattr("src.train.train", fail_if_called)
    with pytest.raises(SystemExit) as exited:
        train_main(["--k", value])
    assert exited.value.code == 2
    assert "--k" in capsys.readouterr().err


def test_cli_experimental_k_never_targets_official_outputs(monkeypatch):
    experimental_k = next(k for k in range(2, K_MAX + 1) if k != FINAL_K)
    received = {}

    def fake_train(data_path, k_override, reports_dir, model_path):
        received.update(k=k_override, reports_dir=reports_dir, model_path=model_path)
        raise ValueError("interrompido pelo teste")

    monkeypatch.setattr("src.train.train", fake_train)
    assert train_main(["--k", str(experimental_k)]) == 1
    expected_dir = EXPERIMENTS_DIR / f"k{experimental_k}"
    assert received["k"] == experimental_k
    assert received["reports_dir"] == expected_dir / "reports"
    assert received["model_path"] == expected_dir / "kmeans_model.joblib"
    assert received["reports_dir"] != REPORTS_DIR and received["model_path"] != MODEL_PATH


@needs_real_data
def test_real_pipeline_end_to_end(tmp_path):
    """Treina na base real com o K final, para validar a consistência no dado efetivo."""
    model_path = tmp_path / "modelo.joblib"
    train(
        k_override=FINAL_K,
        reports_dir=tmp_path / "reports",
        model_path=model_path,
        n_bootstrap=0,
    )
    artifact = load_artifact(model_path)
    pipeline = artifact["pipeline"]
    df = load_dataset(DATA_PATH)
    features = select_features(df)
    kmeans = pipeline.named_steps["kmeans"]

    assert kmeans.labels_.shape == (len(df),)
    np.testing.assert_array_equal(pipeline.predict(features), kmeans.labels_)

    filled = features[NUMERIC_FEATURES].fillna(features[NUMERIC_FEATURES].median())
    chas = features["CHAS"].fillna(imputation_values(pipeline)["CHAS"])
    for profile in denormalize_centroids(pipeline):
        members = kmeans.labels_ == profile["cluster"]
        for feature in NUMERIC_FEATURES:
            assert profile["numeric"][feature] == pytest.approx(filled.loc[members, feature].mean())
        assert profile["CHAS"]["proporcoes"][1] == pytest.approx((chas[members] == 1).mean())
