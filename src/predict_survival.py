"""Task 3: Survival vs Death prediction.

Usage:
    python src/predict_survival.py --test_dir data/test

This script prints predictions directly to the terminal and does not create
an output file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import build_matrix, choose_model_family, discover_expression_files, predict_proba_model


def _select_rows(table: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows and max_rows > 0:
        return table.head(max_rows)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: predict Survival or Death from tumor expression TSV files.")
    parser.add_argument("--test_dir", default="data/test", help="Directory containing evaluation expression TSV files.")
    parser.add_argument("--model_path", default="models/lung_cancer_models.joblib", help="Pretrained model artifact path.")
    parser.add_argument(
        "--model_family",
        default="auto",
        choices=("auto", "rna_protein", "rna", "protein"),
        help="Model family to use. auto uses rna_protein when both RNA and protein files are available.",
    )
    parser.add_argument("--max_rows", type=int, default=0, help="Print only the first N rows. 0 means print all rows.")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    records = discover_expression_files(test_dir)
    if not records:
        raise FileNotFoundError(f"No expression TSV files found in {test_dir}")

    available_modalities = sorted({record["modality"] for record in records})
    bundle = joblib.load(model_path)
    family_name = choose_model_family(bundle, available_modalities, requested=args.model_family)
    family = bundle["families"][family_name]
    modalities = ("rna", "protein") if family_name == "rna_protein" else (family_name,)

    x, meta = build_matrix(records, tissues=("tumor",), modalities=modalities)
    if x.empty:
        raise FileNotFoundError("No tumor expression files were found for the selected modality.")

    prediction, prob = predict_proba_model(family["survival"], x)
    prob_death = prob["prob_Death"] if "prob_Death" in prob.columns else pd.Series([float("nan")] * len(prob))
    prob_survival = prob["prob_Survival"] if "prob_Survival" in prob.columns else pd.Series([float("nan")] * len(prob))

    table = pd.DataFrame(
        {
            "sample_id": meta["sample_id"].to_numpy(),
            "cancer_type": meta["source_cancer"].to_numpy(),
            "prediction": prediction,
            "prob_death": prob_death.to_numpy(),
            "prob_survival": prob_survival.to_numpy(),
        }
    )
    table["prob_death"] = table["prob_death"].map(lambda v: f"{float(v):.4f}")
    table["prob_survival"] = table["prob_survival"].map(lambda v: f"{float(v):.4f}")

    print("=== Task 3: Survival Prediction ===")
    print(f"test_dir: {test_dir}")
    print(f"model_family: {family_name}")
    print(f"samples: {len(table)}")
    threshold_rules = family["survival"].get("threshold_rules", {}) or {}
    if "Death" in threshold_rules:
        death_threshold = float(threshold_rules["Death"])
        print(f"decision_rule: Death if prob_death >= {death_threshold:.4f}; otherwise Survival")
    print()
    print("=== 예측 결과 ===")
    print(_select_rows(table, args.max_rows).to_string(index=False))
    if args.max_rows and args.max_rows > 0 and len(table) > args.max_rows:
        print(f"\n... {len(table) - args.max_rows} more rows not shown. Use --max_rows 0 to print all rows.")


if __name__ == "__main__":
    main()
