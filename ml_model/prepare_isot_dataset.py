"""Convert ISOT's True.csv and Fake.csv into this project's training format."""
import argparse
from pathlib import Path

import pandas as pd


def read_isot(path, label):
    frame = pd.read_csv(path)
    required = {"title", "text"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain ISOT's title and text columns.")
    title = frame["title"].fillna("").astype(str).str.strip()
    body = frame["text"].fillna("").astype(str).str.strip()
    text = (title + ". " + body).str.strip(". ")
    return pd.DataFrame({"text": text, "label": label})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/isot",
                        help="Folder containing True.csv and Fake.csv")
    parser.add_argument("--output", default="data/news.csv",
                        help="Combined output CSV")
    args = parser.parse_args()

    source = Path(args.source_dir)
    true_file, fake_file = source / "True.csv", source / "Fake.csv"
    missing = [str(file) for file in (true_file, fake_file) if not file.is_file()]
    if missing:
        raise FileNotFoundError("Missing: " + ", ".join(missing))

    dataset = pd.concat([read_isot(true_file, 1), read_isot(fake_file, 0)], ignore_index=True)
    dataset = dataset[dataset["text"].str.len() >= 30].drop_duplicates(subset="text")
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False, encoding="utf-8")
    print(f"Created {output} with {len(dataset)} articles")
    print(dataset["label"].value_counts().sort_index().rename({0: "fake", 1: "real"}))


if __name__ == "__main__":
    main()
