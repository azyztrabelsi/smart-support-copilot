from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def main() -> None:
    parser = ArgumentParser(description="Show the first five rows of the dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file where the first five rows will be saved.",
    )
    args = parser.parse_args()

    head = pd.read_csv(DATA_PATH).head(5)
    print(head.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        head.to_csv(args.output, index=False)
        print(f"\nSaved five rows to: {args.output}")


if __name__ == "__main__":
    main()