import pandas as pd
import json


def main():
    # Read the transformed CSV
    df = pd.read_csv("data/transformed_global_opinions.csv")

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
    output_path = "data/transformed_global_opinions.json"
    with open(output_path, "w") as f:
        json.dump(records, f, indent=4)

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
