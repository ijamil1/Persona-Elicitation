"""Test that train/test question lists are identical across countries.

Run from project root: python tests/test_questions_same_across_countries.py
"""

import json
import random
from pathlib import Path


def load_data(country, seed):
    """Load and split data for a given country."""
    with open(Path("data/transformed_global_opinions.json")) as f:
        data = json.load(f)

    country_data = [item for item in data if item["country"] == country]

    # Group by consistency_id
    consistency_groups = {}
    for idx, item in enumerate(country_data):
        cid = item["consistency_id"]
        if cid not in consistency_groups:
            consistency_groups[cid] = []
        consistency_groups[cid].append(idx)

    group_ids = list(consistency_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    total_items = len(country_data)
    target_train_items = int(total_items * 0.75)

    train_ids = []
    train_group_ids = []
    for gid in group_ids:
        if len(train_ids) >= target_train_items:
            break
        train_ids.extend(consistency_groups[gid])
        train_group_ids.append(gid)

    test_group_ids = [gid for gid in group_ids if gid not in train_group_ids]
    test_ids = []
    for gid in test_group_ids:
        test_ids.extend(consistency_groups[gid])

    train = [country_data[i] for i in train_ids]
    test = [country_data[i] for i in test_ids]

    return train, test


def test_questions_same_across_countries():
    """Test that all countries have identical train and test question lists."""
    countries = ["France", "Germany", "Japan", "Russia", "United States"]
    seed = 12345

    results = {}
    for country in countries:
        train, test = load_data(country, seed)
        results[country] = {
            "train_questions": [item["question"] for item in train],
            "test_questions": [item["question"] for item in test],
        }

    ref_country = countries[0]
    ref_train = results[ref_country]["train_questions"]
    ref_test = results[ref_country]["test_questions"]

    # Compare train questions
    print("=== Comparing TRAIN question lists ===")
    for country in countries[1:]:
        other_train = results[country]["train_questions"]
        assert ref_train == other_train, (
            f"Train questions differ between {ref_country} and {country}"
        )
        print(f"{ref_country} vs {country}: IDENTICAL (len={len(ref_train)})")

    # Compare test questions
    print()
    print("=== Comparing TEST question lists ===")
    for country in countries[1:]:
        other_test = results[country]["test_questions"]
        assert ref_test == other_test, (
            f"Test questions differ between {ref_country} and {country}"
        )
        print(f"{ref_country} vs {country}: IDENTICAL (len={len(ref_test)})")

    print()
    print("SUCCESS: All countries have identical train and test question lists")
    print(f"  - Countries tested: {', '.join(countries)}")
    print(f"  - Train questions: {len(ref_train)}")
    print(f"  - Test questions: {len(ref_test)}")


if __name__ == "__main__":
    test_questions_same_across_countries()
