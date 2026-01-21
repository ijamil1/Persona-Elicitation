import csv
import ast
import re
import pandas as pd
from collections import defaultdict


def extract_question_text(full_text):
    """Extract just the question text, removing answer options (A., B., etc.)"""
    # Split on newline followed by A. to get just the question part
    match = re.split(r'\nA\.', full_text)
    return match[0].strip()


def main():
    # Step 1: Read global_opinion_data.csv to extract countries and questions
    with open("data/global_opinion_data.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Extract questions (2nd through last column names)
        # The column headers contain question + answer options, extract just the question
        raw_questions = header[1:]
        questions = [extract_question_text(q) for q in raw_questions]

        # Extract all country values (first column of remaining rows)
        countries = []
        for row in reader:
            countries.append(row[0])

    # Normalize questions: strip whitespace and lowercase for matching
    questions_normalized = set(q.strip().lower() for q in questions)
    countries_set = set(countries)

    print(f"Extracted {len(questions_normalized)} questions from global_opinion_data.csv")
    print(f"Extracted {len(countries_set)} countries: {countries_set}")

    # Step 2: Read Anthropic global opinions dataset
    df = pd.read_csv("hf://datasets/Anthropic/llm_global_opinions/data/global_opinions.csv")

    # Keep only the first 3 columns (question, selections, options)
    df = df[["question", "selections", "options"]]

    # Step 3: Filter questions that appear in extracted questions (normalized comparison)
    # Drop row with NaN question (row index 2538 in the Anthropic dataset)
    df = df.dropna(subset=["question"])
    df = df[df["question"].apply(lambda q: q.strip().lower() in questions_normalized)]
    print(f"After filtering questions: {len(df)} rows")

    # Step 4: Parse selections and options columns (they are string representations)
    # selections column contains defaultdict strings like "defaultdict(<class 'list'>, {...})"
    # Replace "<class 'list'>" with "list" so eval can parse it
    def parse_selections(s):
        s = s.replace("<class 'list'>", "list")
        return eval(s)
    df["selections"] = df["selections"].apply(parse_selections)
    df["options"] = df["options"].apply(ast.literal_eval)

    # Step 5: Filter selections to only include countries from global_opinion_data.csv
    def filter_countries(selections_dict):
        return {k: v for k, v in selections_dict.items() if k in countries_set}

    df["selections"] = df["selections"].apply(filter_countries)

    # Step 6: Reformat to question, country, option, percentage schema
    records = []
    for _, row in df.iterrows():
        question = row["question"]
        options = row["options"]
        selections = row["selections"]

        for country, percentages in selections.items():
            for option, percentage in zip(options, percentages):
                records.append({
                    "question": question,
                    "country": country,
                    "option": option,
                    "percentage": percentage
                })

    result_df = pd.DataFrame(records)
    print(f"Reformatted dataframe: {len(result_df)} rows")

    # Step 7: Add binary label column (1 for max percentage per question/country, 0 otherwise)
    result_df["label"] = result_df.groupby(["question", "country"])["percentage"].transform(
        lambda x: (x == x.max()).astype(int)
    )

    # Step 8: Write to CSV
    output_path = "data/transformed_global_opinions.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Wrote output to {output_path}")
    print(f"Final dataframe shape: {result_df.shape}")
    print(f"\nSample rows:")
    print(result_df.head(10))


if __name__ == "__main__":
    main()
