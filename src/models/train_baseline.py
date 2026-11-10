from argparse import ArgumentParser
from pathlib import Path
import json
import re

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def remove_label_tokens(row: pd.Series) -> str:
    text = normalize_text(row["instruction"])
    labels = [str(row["category"]), str(row["intent"]).replace("_", " ")]
    for label in labels:
        text = re.sub(rf"\b{re.escape(label.lower())}\b", " ", text)
    return normalize_text(text)


def split_without_duplicate_leakage(
    data: pd.DataFrame, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = data["instruction"].map(normalize_text)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_indices, test_indices = next(
        splitter.split(data, y=data["category"], groups=groups)
    )
    return data.iloc[train_indices].copy(), data.iloc[test_indices].copy()


def main() -> None:
    parser = ArgumentParser(description="Train a leakage-aware TF-IDF classification baseline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required_columns = {"instruction", "category", "intent"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if args.sample_size is not None:
        if args.sample_size < 2:
            raise ValueError("--sample-size must be at least 2")
        data = data.sample(
            n=min(args.sample_size, len(data)), random_state=args.random_state
        )

    train_data, test_data = split_without_duplicate_leakage(data, args.random_state)
    train_groups = set(train_data["instruction"].map(normalize_text))
    test_groups = set(test_data["instruction"].map(normalize_text))
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Duplicate leakage detected across splits: {len(overlap)} groups")

    train_text = train_data.apply(remove_label_tokens, axis=1)
    test_text = test_data.apply(remove_label_tokens, axis=1)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    train_vectors = vectorizer.fit_transform(train_text)
    test_vectors = vectorizer.transform(test_text)

    model = LogisticRegression(max_iter=1000, random_state=args.random_state)
    model.fit(train_vectors, train_data["category"])
    predictions = model.predict(test_vectors)
    f1_macro = f1_score(test_data["category"], predictions, average="macro")
    report = classification_report(
        test_data["category"], predictions, output_dict=True, zero_division=0
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "rows": int(len(data)),
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "train_test_duplicate_groups": 0,
        "f1_macro": float(f1_macro),
        "classification_report": report,
    }
    (args.report_dir / "classification_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    labels = sorted(data["category"].unique())
    pd.DataFrame(
        confusion_matrix(test_data["category"], predictions, labels=labels),
        index=labels,
        columns=labels,
    ).to_csv(args.report_dir / "classification_confusion_matrix.csv")

    mlflow.set_experiment("Support-Ticket-Classification")
    with mlflow.start_run(run_name="TFIDF_LogReg_LeakageAware"):
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "max_features": 5000,
                "label_tokens_removed": True,
                "duplicate_aware_split": True,
                "sample_size": args.sample_size or "all",
            }
        )
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Rows: {len(data)}")
    print(f"Train rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")
    print("Train/test duplicate groups: 0")
    print(f"F1-macro: {f1_macro:.4f}")
    print(f"Reports saved to: {args.report_dir}")


if __name__ == "__main__":
    main()
