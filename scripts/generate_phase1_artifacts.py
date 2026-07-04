"""Generate phase-1 acceptance artifacts from current CSVs and reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_summary(phones: Path, aligned: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(
        phones,
        usecols=["dataset_source", "alignment_quality", "split", "gold_binary"],
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )
    aligned_df = pd.read_csv(
        aligned,
        usecols=["dataset_source", "alignment_quality", "split", "gold_binary"],
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )
    return source, aligned_df


def build_data_manifest(phones: Path, aligned: Path, output: Path) -> None:
    source, aligned_df = dataset_summary(phones, aligned)
    pass_df = aligned_df[aligned_df["alignment_quality"] == "pass"]
    rows = [
        {
            "file": str(phones),
            "role": "combined_source_table",
            "rows": len(source),
            "dataset_sources": "; ".join(
                f"{k}={v}" for k, v in source["dataset_source"].value_counts().items()
            ),
            "alignment_quality": "; ".join(
                f"{k or 'blank'}={v}" for k, v in source["alignment_quality"].value_counts().items()
            ),
            "train_rows": int((source["split"] == "train").sum()),
            "dev_rows": int((source["split"] == "dev").sum()),
            "test_rows": int((source["split"] == "test").sum()),
            "gold_binary_0": int((source["gold_binary"] == 0).sum()),
            "gold_binary_1": int((source["gold_binary"] == 1).sum()),
        },
        {
            "file": str(aligned),
            "role": "primary_aligned_model_input",
            "rows": len(aligned_df),
            "dataset_sources": "; ".join(
                f"{k}={v}" for k, v in aligned_df["dataset_source"].value_counts().items()
            ),
            "alignment_quality": "; ".join(
                f"{k}={v}" for k, v in aligned_df["alignment_quality"].value_counts().items()
            ),
            "train_rows": int((aligned_df["split"] == "train").sum()),
            "dev_rows": int((aligned_df["split"] == "dev").sum()),
            "test_rows": int((aligned_df["split"] == "test").sum()),
            "gold_binary_0": int((aligned_df["gold_binary"] == 0).sum()),
            "gold_binary_1": int((aligned_df["gold_binary"] == 1).sum()),
        },
        {
            "file": str(aligned),
            "role": "first_round_pass_only_training_input",
            "rows": len(pass_df),
            "dataset_sources": "; ".join(
                f"{k}={v}" for k, v in pass_df["dataset_source"].value_counts().items()
            ),
            "alignment_quality": "pass",
            "train_rows": int((pass_df["split"] == "train").sum()),
            "dev_rows": int((pass_df["split"] == "dev").sum()),
            "test_rows": int((pass_df["split"] == "test").sum()),
            "gold_binary_0": int((pass_df["gold_binary"] == 0).sum()),
            "gold_binary_1": int((pass_df["gold_binary"] == 1).sum()),
        },
    ]
    write_csv(
        output,
        rows,
        [
            "file",
            "role",
            "rows",
            "dataset_sources",
            "alignment_quality",
            "train_rows",
            "dev_rows",
            "test_rows",
            "gold_binary_0",
            "gold_binary_1",
        ],
    )


def build_label_spec(output: Path) -> None:
    write_text(
        output,
        """# Label Specification

## Binary Label

`gold_binary` is the first-round training target.

| value | meaning | use |
|---:|---|---|
| 1 | acceptable/correct target phone realization | positive class |
| 0 | phone error or unacceptable realization | error class |

Current SpeechOcean mapping is already materialized in `phones_aligned.csv`.

## Auxiliary Labels

- `source_score`: original phone-level score where available.
- `attention_binary`: broader attention/needs-review flag.
- `gold_three_class`: correct / acceptable-but-accented / error-style label where available.
- `error_type`: source or derived error type. It is not treated as a first-round hard target.

## First-Round Training Filter

Use only rows with:

```text
alignment_quality == "pass"
```

Rows with `alignment_quality == "review"` are retained for analysis and later model improvement, but are excluded from the first-round training baseline.

## Split Rule

Use the existing `split` column:

```text
train / dev / test
```

Thresholds must be selected on `dev`; final metrics must be reported on `test`.
""",
    )


def build_acceptance_checklist(output: Path) -> None:
    rows = [
        {
            "requirement": "SpeechOcean phone-level aligned data",
            "status": "complete",
            "evidence": "phones_aligned.csv; reports/data_manifest.csv",
        },
        {
            "requirement": "speaker-level split manifest and leakage check",
            "status": "complete",
            "evidence": "reports/split_manifest.csv; reports/speaker_split_check.csv",
        },
        {
            "requirement": "pass-only first-round training filter",
            "status": "complete",
            "evidence": "run_* scripts --alignment-quality pass; README.md",
        },
        {
            "requirement": "majority-class baseline",
            "status": "complete",
            "evidence": "reports/majority_baseline_metrics.csv",
        },
        {
            "requirement": "feature baseline classifier",
            "status": "complete",
            "evidence": "models/*.joblib; reports/feature_baseline_metrics.csv",
        },
        {
            "requirement": "GOP or equivalent acoustic evidence baseline",
            "status": "partial",
            "evidence": "proxy_gop_score exists; true acoustic GOP is not implemented",
        },
        {
            "requirement": "phone/phone-group threshold calibration",
            "status": "partial",
            "evidence": "reports/proxy_gop_group_thresholds.csv; target_phone thresholds not yet implemented",
        },
        {
            "requirement": "self-supervised speech representation model",
            "status": "not_started",
            "evidence": "No wav2vec2/HuBERT/WavLM/XLS-R embedding pipeline yet",
        },
        {
            "requirement": "fusion model",
            "status": "not_started",
            "evidence": "No GOP + SSL + tabular feature fusion model yet",
        },
        {
            "requirement": "formal metrics and confusion matrix",
            "status": "complete",
            "evidence": "reports/formal_eval_metrics.csv; reports/formal_eval_confusion_matrix.csv",
        },
        {
            "requirement": "100 utterance prediction samples",
            "status": "complete",
            "evidence": "reports/prediction_samples_100_utterances.csv",
        },
        {
            "requirement": "error analysis",
            "status": "complete",
            "evidence": "reports/error_analysis_summary.md; reports/error_cases.csv",
        },
        {
            "requirement": "reproducibility materials",
            "status": "complete",
            "evidence": "README.md; requirements.txt; scripts/",
        },
        {
            "requirement": "target model metrics",
            "status": "not_met",
            "evidence": "Best current test BA=0.604845, Macro-F1=0.384813, AUC=0.645927",
        },
    ]
    write_csv(output, rows, ["requirement", "status", "evidence"])


def best_model_line(comparison: Path) -> str:
    if not comparison.exists():
        return "No model comparison report found."
    text = comparison.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "Best test model" in line:
            return line.lstrip("- ").strip()
    return "See reports/model_comparison.md."


def build_phase1_report(output: Path, comparison: Path) -> None:
    write_text(
        output,
        f"""# Phase 1 Model Design Status

## Summary

The project now contains a reproducible first-round baseline for phone-level pronunciation correctness. It satisfies the runnable baseline/evaluation part of the phase-1 plan, but it does not yet satisfy the full model-design target because true acoustic GOP, self-supervised speech embeddings, and fusion modeling are still missing.

## Current Best Model

{best_model_line(comparison)}

## Completed

- SpeechOcean phone-level aligned input is available in `phones_aligned.csv`.
- First-round model training is restricted to `alignment_quality == "pass"`.
- Majority-class, logistic regression, and random forest baselines are implemented.
- Proxy GOP-style score and phone-group threshold calibration are implemented.
- Formal test metrics, confusion matrices, 100-utterance prediction samples, model comparison, and error analysis are generated.
- Reproducibility instructions are documented in `README.md`.

## Partial Or Missing

- True acoustic GOP is not implemented. Current `proxy_gop_score` is a model probability, not a likelihood/posterior GOP score.
- Self-supervised representation models such as wav2vec2, HuBERT, WavLM, or XLS-R are not implemented.
- Fusion of GOP, duration, phone identity, phone group, and self-supervised embeddings is not implemented.
- Current metrics do not meet the target line from the plan.
- `review` alignment rows are retained but excluded from first-round training.

## Recommended Next Iteration

1. Add real acoustic features or GOP scores from an acoustic model.
2. Add per-target-phone thresholds with fallback to phone-group thresholds.
3. Add a frozen self-supervised embedding extractor once model files and audio paths are available.
4. Re-run comparison with majority baseline, GOP baseline, tabular model, SSL model, and fusion model.
5. Focus error reduction on vowels, fricatives, stops, nasals, and liquids based on `reports/error_analysis_summary.md`.
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phones", type=Path, default=Path("phones.csv"))
    parser.add_argument("--aligned", type=Path, default=Path("phones_aligned.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    build_data_manifest(args.phones, args.aligned, args.reports_dir / "data_manifest.csv")
    build_label_spec(args.reports_dir / "label_spec.md")
    build_acceptance_checklist(args.reports_dir / "phase1_acceptance_checklist.csv")
    build_phase1_report(
        args.reports_dir / "phase1_model_design_status.md",
        args.reports_dir / "model_comparison.md",
    )
    print(f"Wrote phase-1 artifacts to {args.reports_dir}")


if __name__ == "__main__":
    main()
