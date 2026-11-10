from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "customer_support_ticket_records.csv"
)
SAFE_NUMERIC_FEATURES = ["resolution_time_hours", "customer_satisfaction"]


def find_volume_spikes(
    data: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    daily_counts = (
        data.groupby([data["created_at"].dt.date, "issue_type"])
        .size()
        .rename("ticket_count")
        .reset_index()
        .rename(columns={"created_at": "date"})
    )
    daily_counts["date"] = pd.to_datetime(daily_counts["date"])
    daily_counts = daily_counts.sort_values(["issue_type", "date"])
    grouped = daily_counts.groupby("issue_type", group_keys=False)
    daily_counts["baseline_mean"] = grouped["ticket_count"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=3).mean()
    )
    daily_counts["baseline_std"] = grouped["ticket_count"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=3).std()
    )
    daily_counts["spike_score"] = (
        daily_counts["ticket_count"] - daily_counts["baseline_mean"]
    ) / daily_counts["baseline_std"].replace(0, pd.NA)
    return daily_counts[
        daily_counts["spike_score"].ge(threshold)
        & daily_counts["baseline_mean"].notna()
    ].sort_values("spike_score", ascending=False)


def find_operational_outliers(data: pd.DataFrame, contamination: float) -> pd.DataFrame:
    features = data[SAFE_NUMERIC_FEATURES].copy()
    detector = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        ),
    )
    predictions = detector.fit_predict(features)
    scores = detector[-1].decision_function(
        detector[:-1].transform(features)
    )
    outliers = data.loc[predictions == -1, [
        "created_at",
        "issue_type",
        "priority",
        "status",
        "channel",
        *SAFE_NUMERIC_FEATURES,
    ]].copy()
    outliers["outlier_score"] = scores[predictions == -1]
    return outliers.sort_values("outlier_score")


def main() -> None:
    parser = ArgumentParser(description="Analyze support-ticket volume and operational anomalies.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--spike-threshold", type=float, default=3.0)
    parser.add_argument("--contamination", type=float, default=0.01)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required_columns = {
        "created_at",
        "issue_type",
        "priority",
        "status",
        "channel",
        *SAFE_NUMERIC_FEATURES,
    }
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce")
    valid_data = data.dropna(subset=["created_at"]).copy()
    spikes = find_volume_spikes(valid_data, args.spike_threshold)
    outliers = find_operational_outliers(valid_data, args.contamination)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    spikes.to_csv(args.report_dir / "ticket_volume_spikes.csv", index=False)
    outliers.to_csv(args.report_dir / "ticket_operational_outliers.csv", index=False)
    summary = {
        "rows": int(len(data)),
        "valid_timestamp_rows": int(len(valid_data)),
        "invalid_timestamp_rows": int(len(data) - len(valid_data)),
        "issue_types": int(valid_data["issue_type"].nunique()),
        "volume_spikes": int(len(spikes)),
        "operational_outliers": int(len(outliers)),
        "excluded_identifier_columns": [
            "ticket_id",
            "customer_id",
            "phone_number",
            "assigned_to",
        ],
        "volume_spike_threshold": args.spike_threshold,
        "outlier_contamination": args.contamination,
    }
    (args.report_dir / "ticket_anomaly_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Rows: {summary['rows']}")
    print(f"Valid timestamp rows: {summary['valid_timestamp_rows']}")
    print(f"Volume spikes: {summary['volume_spikes']}")
    print(f"Operational outliers: {summary['operational_outliers']}")
    print(f"Reports saved to: {args.report_dir}")


if __name__ == "__main__":
    main()