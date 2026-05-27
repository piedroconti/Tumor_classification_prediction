"""Train pretrained models from the public training TSV files.

This script is for the project author. The professor/evaluator only needs
src/predict.py and models/lung_cancer_models.joblib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from common import build_matrix, discover_expression_files, read_survival_labels


RANDOM_STATE = 42


def _safe_median_impute(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(arr, axis=0).astype(np.float32)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    missing = np.where(np.isnan(arr))
    if len(missing[0]) > 0:
        arr[missing] = medians[missing[1]]
    return arr, medians


def _welch_score(arr: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    mask0 = y == classes[0]
    mask1 = y == classes[1]
    n0 = max(int(mask0.sum()), 1)
    n1 = max(int(mask1.sum()), 1)
    mean0 = arr[mask0].mean(axis=0)
    mean1 = arr[mask1].mean(axis=0)
    var0 = arr[mask0].var(axis=0, ddof=1)
    var1 = arr[mask1].var(axis=0, ddof=1)
    score = np.abs(mean1 - mean0) / np.sqrt(var0 / n0 + var1 / n1 + 1e-8)
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def make_estimators(kind: str) -> List[Tuple[str, object]]:
    """Create task-specific estimator ensembles.

    Linear models are used because the data are high-dimensional and the
    number of patients is small. This is usually more stable than a very
    flexible tree model for omics data with thousands of features.
    """
    if kind in {"tumor_normal", "luad_lscc"}:
        return [
            (
                "logistic_l2_balanced_c1",
                LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE),
            ),
            (
                "logistic_l2_c1",
                LogisticRegression(max_iter=5000, C=1.0, class_weight=None, solver="liblinear", random_state=RANDOM_STATE + 1),
            ),
            (
                "logistic_l2_balanced_c03",
                LogisticRegression(max_iter=5000, C=0.3, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE + 2),
            ),
            (
                "logistic_l2_c03",
                LogisticRegression(max_iter=5000, C=0.3, class_weight=None, solver="liblinear", random_state=RANDOM_STATE + 3),
            ),
        ]

    if kind == "survival":
        return [
            (
                "logistic_l2_c1",
                LogisticRegression(max_iter=5000, C=1.0, class_weight=None, solver="liblinear", random_state=RANDOM_STATE),
            ),
            (
                "logistic_l2_c03",
                LogisticRegression(max_iter=5000, C=0.3, class_weight=None, solver="liblinear", random_state=RANDOM_STATE + 1),
            ),
            (
                "logistic_l2_balanced_c1",
                LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE + 2),
            ),
        ]

    raise ValueError(f"Unknown estimator kind: {kind}")

def fit_model(
    x: pd.DataFrame,
    y: Sequence[str],
    task: str,
    model_family: str,
    top_k: int,
    estimator_kind: str,
    missing_threshold: float = 0.50,
    threshold_rules: Dict[str, float] | None = None,
) -> Dict:
    y_arr = np.asarray(y, dtype=str)
    classes = np.asarray(sorted(pd.unique(y_arr).tolist()), dtype=object)
    if len(classes) != 2:
        raise ValueError(f"Only binary classification is supported, got classes={classes}")

    feature_names = np.asarray(x.columns.astype(str), dtype=object)
    arr = x.to_numpy(dtype=np.float32, copy=True)

    missing_rate = np.mean(np.isnan(arr), axis=0)
    keep = missing_rate <= missing_threshold
    arr = arr[:, keep]
    feature_names = feature_names[keep]

    arr, medians = _safe_median_impute(arr)

    scores = _welch_score(arr, y_arr, classes)
    k = min(int(top_k), arr.shape[1])
    selected = np.argsort(scores)[::-1][:k]

    selected_arr = arr[:, selected]
    selected_features = feature_names[selected]
    selected_medians = medians[selected]
    selected_scores = scores[selected].astype(np.float32)

    mean = selected_arr.mean(axis=0).astype(np.float32)
    std = selected_arr.std(axis=0, ddof=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    z = ((selected_arr - mean) / std).astype(np.float32)

    estimators = make_estimators(estimator_kind)
    fitted_estimators: List[Tuple[str, object]] = []
    for name, estimator in estimators:
        estimator.fit(z, y_arr)
        fitted_estimators.append((name, estimator))

    # Weight linear models slightly more on high-dimensional omics data.
    if estimator_kind in {"tumor_normal", "luad_lscc"}:
        weights = np.asarray([0.30, 0.25, 0.25, 0.20], dtype=np.float64)
    else:
        weights = np.asarray([0.45, 0.35, 0.20], dtype=np.float64)
    weights = weights / weights.sum()

    model = {
        "task": task,
        "model_family": model_family,
        "features": selected_features,
        "medians": selected_medians,
        "mean": mean,
        "std": std,
        "classes": classes,
        "feature_scores": selected_scores,
        "estimators": fitted_estimators,
        "weights": weights,
        "threshold_rules": threshold_rules or {},
        "class_counts": pd.Series(y_arr).value_counts().sort_index().to_dict(),
        "top_k": int(k),
        "missing_threshold": float(missing_threshold),
    }
    return model


def predict_training(model: Dict, x: pd.DataFrame) -> np.ndarray:
    # Local import avoids circular import during script execution.
    from common import predict_proba_model

    pred, _ = predict_proba_model(model, x)
    return pred


def train_family(train_dir: Path, records: List[Dict[str, str]], family_name: str, modalities: Sequence[str]) -> Dict:
    print(f"\n=== Training model family: {family_name} ({', '.join(modalities)}) ===")
    family: Dict[str, Dict] = {}
    sanity: Dict[str, float] = {}

    # Task 1: Tumor vs Normal
    x_tn, meta_tn = build_matrix(records, tissues=("tumor", "nat"), modalities=modalities)
    y_tn = np.where(meta_tn["source_tissue"].to_numpy() == "tumor", "Tumor", "Normal")
    k_tn = 5000 if family_name == "rna_protein" else 3000
    family["tumor_normal"] = fit_model(
        x_tn, y_tn, task="tumor_normal", model_family=family_name, top_k=k_tn, estimator_kind="tumor_normal"
    )
    sanity["tumor_normal_training_accuracy"] = float(accuracy_score(y_tn, predict_training(family["tumor_normal"], x_tn)))
    print(f"Task1 sanity accuracy: {sanity['tumor_normal_training_accuracy']:.4f}")

    # Task 2: LUAD vs LSCC
    x_ca, meta_ca = build_matrix(records, tissues=("tumor",), modalities=modalities)
    y_ca = meta_ca["source_cancer"].to_numpy(dtype=str)
    k_ca = 5000 if family_name == "rna_protein" else 3000
    family["luad_lscc"] = fit_model(
        x_ca, y_ca, task="luad_lscc", model_family=family_name, top_k=k_ca, estimator_kind="luad_lscc"
    )
    sanity["luad_lscc_training_accuracy"] = float(accuracy_score(y_ca, predict_training(family["luad_lscc"], x_ca)))
    print(f"Task2 sanity accuracy: {sanity['luad_lscc_training_accuracy']:.4f}")

    # Task 3: Survival vs Death, combined LUAD+LSCC tumor model.
    x_sv, meta_sv = build_matrix(records, tissues=("tumor",), modalities=modalities)
    y_sv = read_survival_labels(train_dir, meta_sv)
    k_sv = 80 if family_name == "rna_protein" else 60
    family["survival"] = fit_model(
        x_sv,
        y_sv,
        task="survival",
        model_family=family_name,
        top_k=k_sv,
        estimator_kind="survival",
        threshold_rules={"Death": 0.95},
    )
    sanity["survival_training_accuracy"] = float(accuracy_score(y_sv, predict_training(family["survival"], x_sv)))
    print(f"Task3 sanity accuracy: {sanity['survival_training_accuracy']:.4f}")
    family["sanity"] = sanity
    return family


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lung cancer prediction models.")
    parser.add_argument("--train_dir", default="data/public_train", help="Directory containing public training TSV files.")
    parser.add_argument("--model_path", default="models/lung_cancer_models.joblib", help="Output model artifact path.")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    records = discover_expression_files(train_dir)
    if not records:
        raise FileNotFoundError(f"No expression TSV files found in {train_dir}")

    families_to_train = {
        "rna_protein": ("rna", "protein"),
        "rna": ("rna",),
        "protein": ("protein",),
    }

    bundle = {
        "schema_version": "2.0-high-accuracy",
        "description": "Pretrained models for LUAD/LSCC tumor-normal, cancer-type, and survival prediction.",
        "families": {},
    }

    for family_name, modalities in families_to_train.items():
        available = {record["modality"] for record in records}
        if set(modalities).issubset(available):
            bundle["families"][family_name] = train_family(train_dir, records, family_name, modalities)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    print(f"\nSaved model artifact to {model_path}")

    summary = {
        family: bundle["families"][family].get("sanity", {})
        for family in bundle["families"]
    }
    summary_path = model_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
