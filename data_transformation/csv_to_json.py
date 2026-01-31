import argparse
import pandas as pd
import json


def main(dataset="all"):
    if dataset == "binary":
        csv_path = "data/transformed_global_opinions_binary.csv"
        output_path = "data/transformed_global_opinions_binary.json"
    else:
        csv_path = "data/transformed_global_opinions.csv"
        output_path = "data/transformed_global_opinions.json"

    # Read the transformed CSV
    df = pd.read_csv(csv_path)

    # Create a mapping of (question, country) pairs to consistency_ids
    unique_pairs = df[["question", "country"]].drop_duplicates()
    pair_to_id = {
        (row["question"], row["country"]): idx
        for idx, row in unique_pairs.iterrows()
    }

    # Build the list of JSON objects
    records = []
    for _, row in df.iterrows():
        records.append({
            "question": row["question"],
            "country": row["country"],
            "choice": row["option"],
            "label": int(row["label"]),
            "consistency_id": pair_to_id[(row["question"], row["country"])]
        })

    # Write to JSON file
    with open(output_path, "w") as f:
        json.dump(records, f, indent=4)

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "binary"],
                        help="'all' for transformed_global_opinions, 'binary' for binary-only questions")
    args = parser.parse_args()
    main(dataset=args.dataset)
