"""Common utilities for lung cancer prediction project.

This module intentionally keeps the data interface simple:
- input TSV files are expression matrices with rows = genes/features and columns = samples
- model artifacts are saved as dictionaries by joblib
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CANCERS = ("LUAD", "LSCC")
MODALITIES = ("rna", "protein")
TISSUES = ("tumor", "nat")


def discover_expression_files(data_dir: str | Path) -> List[Dict[str, str]]:
    """Find expression TSV files and infer cancer/modality/tissue from filenames.

    Accepted examples:
    - LUAD_trainingset_rna_expression_tumor.tsv
    - LSCC_testset_protein_expression_nat.tsv
    - any filename containing LUAD/LSCC, rna/protein, expression, tumor/nat.
    """
    data_dir = Path(data_dir)
    records: List[Dict[str, str]] = []
    pattern = re.compile(r"(LUAD|LSCC).*?(rna|protein)_expression_(tumor|nat).*?\.tsv$", re.I)
    for path in sorted(data_dir.glob("*.tsv")):
        match = pattern.search(path.name)
        if match:
            records.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "cancer": match.group(1).upper(),
                    "modality": match.group(2).lower(),
                    "tissue": match.group(3).lower(),
                }
            )
    return records


def discover_survival_files(data_dir: str | Path) -> Dict[str, Path]:
    """Find survival TSV files by cancer name."""
    data_dir = Path(data_dir)
    out: Dict[str, Path] = {}
    for cancer in CANCERS:
        matches = sorted(data_dir.glob(f"{cancer}*overall_survival*.tsv"))
        if matches:
            out[cancer] = matches[0]
    return out


def read_expression(path: str | Path, modality: str) -> pd.DataFrame:
    """Read one expression TSV and return sample x feature matrix.

    Original TSV format is feature x sample. The first column stores feature IDs.
    Returned column names are prefixed with modality, for example:
    rna:ENSG00000000003.15, protein:ENSG00000000003.15
    """
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] < 2:
        raise ValueError(f"Expression file has too few columns: {path}")

    feature_ids = df.iloc[:, 0].astype(str).to_numpy()
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    matrix = values.T
    matrix.columns = [f"{modality}:{feature_id}" for feature_id in feature_ids]
    matrix.index = values.columns.astype(str)
    return matrix.astype(np.float32, copy=False)


def build_matrix(
    records: Sequence[Dict[str, str]],
    cancers: Optional[Iterable[str]] = None,
    tissues: Optional[Iterable[str]] = None,
    modalities: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a single sample x feature matrix from expression records.

    Samples are indexed as cancer|tissue|sample_id to avoid collisions between
    tumor and NAT samples from the same patient.
    """
    cancer_set = {c.upper() for c in cancers} if cancers else None
    tissue_set = {t.lower() for t in tissues} if tissues else None
    modality_set = {m.lower() for m in modalities} if modalities else set(MODALITIES)

    dfs_by_modality: Dict[str, List[pd.DataFrame]] = {m: [] for m in modality_set}
    meta_rows: List[Tuple[str, str, str, str, str]] = []

    for record in records:
        cancer = record["cancer"].upper()
        tissue = record["tissue"].lower()
        modality = record["modality"].lower()

        if cancer_set is not None and cancer not in cancer_set:
            continue
        if tissue_set is not None and tissue not in tissue_set:
            continue
        if modality not in modality_set:
            continue

        matrix = read_expression(record["path"], modality)
        keys = [f"{cancer}|{tissue}|{sample_id}" for sample_id in matrix.index]
        for key, sample_id in zip(keys, matrix.index):
            meta_rows.append((key, str(sample_id), cancer, tissue, record["name"]))
        matrix.index = keys
        dfs_by_modality[modality].append(matrix)

    matrices: List[pd.DataFrame] = []
    for modality in sorted(dfs_by_modality):
        if dfs_by_modality[modality]:
            matrices.append(pd.concat(dfs_by_modality[modality], axis=0, join="outer", sort=False))

    if not matrices:
        return pd.DataFrame(), pd.DataFrame()

    x = pd.concat(matrices, axis=1, join="outer", sort=False)
    x = x.loc[:, ~x.columns.duplicated()].copy()

    meta = (
        pd.DataFrame(meta_rows, columns=["key", "sample_id", "source_cancer", "source_tissue", "source_file"])
        .drop_duplicates("key")
        .set_index("key")
        .loc[x.index]
    )
    return x, meta


def read_survival_labels(data_dir: str | Path, meta: pd.DataFrame) -> np.ndarray:
    """Return Survival/Death labels aligned to meta rows."""
    survival_files = discover_survival_files(data_dir)
    labels: List[str] = []
    cache: Dict[str, pd.DataFrame] = {}

    for _, row in meta.iterrows():
        cancer = row["source_cancer"]
        sample_id = row["sample_id"]
        if cancer not in survival_files:
            raise FileNotFoundError(f"Missing survival file for {cancer} in {data_dir}")
        if cancer not in cache:
            cache[cancer] = pd.read_csv(survival_files[cancer], sep="\t").set_index("case_id")
        survival = cache[cancer]
        if sample_id not in survival.index:
            raise KeyError(f"Sample {sample_id} not found in {survival_files[cancer]}")
        event = int(survival.loc[sample_id, "OS_event"])
        labels.append("Death" if event == 1 else "Survival")
    return np.asarray(labels, dtype=str)


def choose_model_family(model_bundle: Dict, available_modalities: Iterable[str], requested: str = "auto") -> str:
    """Choose rna_protein/rna/protein model family."""
    families = set(model_bundle["families"].keys())
    available = {m.lower() for m in available_modalities}

    if requested != "auto":
        if requested not in families:
            raise ValueError(f"Requested model family {requested!r} is not available. Available: {sorted(families)}")
        return requested

    if {"rna", "protein"}.issubset(available) and "rna_protein" in families:
        return "rna_protein"
    if "rna" in available and "rna" in families:
        return "rna"
    if "protein" in available and "protein" in families:
        return "protein"
    # Fallback to the first saved family.
    return sorted(families)[0]


def transform_for_model(model: Dict, x: pd.DataFrame) -> np.ndarray:
    """Apply saved feature alignment, median imputation, and z-score scaling."""
    features = list(model["features"])
    arr = x.reindex(columns=features).to_numpy(dtype=np.float32, copy=True)

    medians = np.asarray(model["medians"], dtype=np.float32)
    missing = np.where(np.isnan(arr))
    if len(missing[0]) > 0:
        arr[missing] = medians[missing[1]]

    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32, copy=False)


def predict_proba_model(model: Dict, x: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Predict labels and probabilities using an ensemble model dictionary."""
    z = transform_for_model(model, x)
    classes = list(model["classes"])
    prob = np.zeros((z.shape[0], len(classes)), dtype=np.float64)

    estimators = model["estimators"]
    weights = np.asarray(model.get("weights", np.ones(len(estimators)) / len(estimators)), dtype=np.float64)
    weights = weights / weights.sum()

    for weight, (name, estimator) in zip(weights, estimators):
        estimator_prob = estimator.predict_proba(z)
        aligned = np.zeros_like(prob)
        for i, cls in enumerate(estimator.classes_):
            aligned[:, classes.index(cls)] = estimator_prob[:, i]
        prob += weight * aligned

    # Default argmax prediction.
    pred = np.asarray(classes, dtype=object)[np.argmax(prob, axis=1)]

    # Optional conservative threshold rule for imbalanced survival prediction.
    threshold_rules = model.get("threshold_rules", {}) or {}
    if "Death" in threshold_rules and "Death" in classes and "Survival" in classes:
        death_idx = classes.index("Death")
        threshold = float(threshold_rules["Death"])
        pred = np.where(prob[:, death_idx] >= threshold, "Death", "Survival")

    prob_df = pd.DataFrame(prob, index=x.index, columns=[f"prob_{cls}" for cls in classes])
    return pred.astype(str), prob_df


def write_prediction_files(
    out_dir: str | Path,
    task_name: str,
    csv_name: str,
    tsv_name: str,
    table: pd.DataFrame,
) -> None:
    """Write both CSV and TSV for compatibility with different grading scripts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / csv_name, index=False)
    table.to_csv(out_dir / tsv_name, sep="\t", index=False)
    print(f"[{task_name}] wrote {out_dir / csv_name}")
    print(f"[{task_name}] wrote {out_dir / tsv_name}")
