"""Sanity-check the pretrained models using public training TSVs as pseudo-test input.

This script is for the project author, not for the evaluator. It does not write
prediction files; it prints training-data sanity accuracy directly to terminal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import build_matrix, choose_model_family, discover_expression_files, predict_proba_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Public-training sanity accuracy check.")
    parser.add_argument("--data_dir", default="data/public_train", help="Directory containing the public training TSV files.")
    parser.add_argument("--model_path", default="models/lung_cancer_models.joblib", help="Pretrained model artifact path.")
    parser.add_argument(
        "--model_family",
        default="auto",
        choices=("auto", "rna_protein", "rna", "protein"),
        help="Model family to use for the sanity check.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)

    records = discover_expression_files(data_dir)
    if not records:
        raise FileNotFoundError(f"No expression TSV files found in {data_dir}")

    bundle = joblib.load(model_path)
    available_modalities = sorted({record["modality"] for record in records})
    family_name = choose_model_family(bundle, available_modalities, requested=args.model_family)
    family = bundle["families"][family_name]
    modalities = ("rna", "protein") if family_name == "rna_protein" else (family_name,)

    # Task 1: Tumor vs Normal
    x_tn, meta_tn = build_matrix(records, tissues=("tumor", "nat"), modalities=modalities)
    pred_tn, _ = predict_proba_model(family["tumor_normal"], x_tn)
    answer_tn = meta_tn["source_tissue"].map({"tumor": "Tumor", "nat": "Normal"}).to_numpy()
    acc_tn = (pred_tn == answer_tn).mean()

    # Task 2: LUAD vs LSCC
    x_ca, meta_ca = build_matrix(records, tissues=("tumor",), modalities=modalities)
    pred_ca, _ = predict_proba_model(family["luad_lscc"], x_ca)
    answer_ca = meta_ca["source_cancer"].to_numpy()
    acc_ca = (pred_ca == answer_ca).mean()

    # Task 3: Survival vs Death
    pred_sv, _ = predict_proba_model(family["survival"], x_ca)
    survival_answers = []
    survival_cache = {}
    for _, row in meta_ca.iterrows():
        cancer = row["source_cancer"]
        sample_id = row["sample_id"]
        if cancer not in survival_cache:
            path = data_dir / f"{cancer}_trainingset_overall_survival.tsv"
            survival_cache[cancer] = pd.read_csv(path, sep="\t").set_index("case_id")
        event = int(survival_cache[cancer].loc[sample_id, "OS_event"])
        survival_answers.append("Death" if event == 1 else "Survival")
    acc_sv = (pred_sv == pd.Series(survival_answers).to_numpy()).mean()

    print("=== Public Training Sanity Check ===")
    print(f"data_dir: {data_dir}")
    print(f"model_family: {family_name}")
    print(f"Task 1 Tumor vs Normal accuracy: {acc_tn:.4f}")
    print(f"Task 2 LUAD vs LSCC accuracy:    {acc_ca:.4f}")
    print(f"Task 3 Survival accuracy:        {acc_sv:.4f}")
    print()
    print("Note: this is a sanity check on public training data, not hidden-test performance.")


if __name__ == "__main__":
    main()
