from argparse import ArgumentParser
from pathlib import Path
import re

import pandas as pd


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def audit_dataset(data_path: Path) -> dict[str, object]:
    data = pd.read_csv(data_path)
    required_columns = {"instruction", "category", "intent", "response"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    normalized_instruction = data["instruction"].map(normalize_text)
    duplicate_mask = normalized_instruction.duplicated(keep=False)
    duplicate_groups = int(normalized_instruction[duplicate_mask].nunique())
    conflicting_duplicate_groups = int(
        data.loc[duplicate_mask]
        .assign(_normalized_instruction=normalized_instruction[duplicate_mask])
        .groupby("_normalized_instruction", dropna=False)["category"]
        .nunique()
        .gt(1)
        .sum()
    )

    category_token_matches = int(
        sum(
            bool(re.search(rf"\b{re.escape(str(category).lower())}\b", text))
            for text, category in zip(normalized_instruction, data["category"])
        )
    )
    intent_token_matches = int(
        sum(
            bool(re.search(rf"\b{re.escape(str(intent).lower())}\b", text))
            for text, intent in zip(normalized_instruction, data["intent"])
        )
    )

    return {
        "rows": int(len(data)),
        "columns": list(data.columns),
        "missing_values": {column: int(value) for column, value in data.isna().sum().items()},
        "category_distribution": {
            str(label): int(value) for label, value in data["category"].value_counts().items()
        },
        "exact_duplicate_instruction_rows": int(duplicate_mask.sum()),
        "duplicate_instruction_groups": duplicate_groups,
        "conflicting_duplicate_groups": conflicting_duplicate_groups,
        "instructions_containing_category_token": category_token_matches,
        "instructions_containing_intent_token": intent_token_matches,
        "response_is_not_used_as_model_input": True,
    }


def main() -> None:
    parser = ArgumentParser(description="Audit the Bitext dataset for leakage risks.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    report = audit_dataset(args.data)
    for key, value in report.items():
        print(f"{key}: {value}")

    if args.output:
        import json

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to: {args.output}")


if __name__ == "__main__":
    main()