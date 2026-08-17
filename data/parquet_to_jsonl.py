import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input parquet file")
    parser.add_argument("output", help="Output jsonl file")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    df.to_json(
        args.output,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print(f"Converted {len(df)} rows")
    print(f"{args.input} -> {args.output}")


if __name__ == "__main__":
    main()