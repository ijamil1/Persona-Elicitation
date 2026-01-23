"""Test that load_data returns consistent outputs across multiple calls.

This script extracts the load_data logic to avoid heavy dependencies.
"""

import json
import random
from pathlib import Path


def get_root_directory():
    """Get the root directory of the project."""
    return Path(__file__).parent


def load_data(args):
    """Copy of load_data from ICM.py for testing."""
    with open(get_root_directory() / "data/transformed_global_opinions.json") as f:
        data = json.load(f)

    # Filter by country
    country_data = [item for item in data if item['country'] == args.country]

    template = """Question: {question}
        Claim: {answer}
        I think the claim is """

    for i in country_data:
        i['prompt'] = template.format(question=i['question'], answer=i['choice'])

    # Group by consistency_id
    consistency_groups = {}
    for idx, item in enumerate(country_data):
        cid = item['consistency_id']
        if cid not in consistency_groups:
            consistency_groups[cid] = []
        consistency_groups[cid].append(idx)

    # Shuffle groups and accumulate until we reach ~75% of total items
    group_ids = list(consistency_groups.keys())
    # Use a seeded RNG to ensure deterministic train/test splits across calls
    rng = random.Random(args.seed)
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

    # Remaining groups go to test
    test_group_ids = [gid for gid in group_ids if gid not in train_group_ids]
    test_ids = []
    for gid in test_group_ids:
        test_ids.extend(consistency_groups[gid])

    # Build train and test lists
    train = [country_data[i] for i in train_ids]
    test = [country_data[i] for i in test_ids]

    # fewshot_ids is just all indices into train
    fewshot_ids = list(range(len(train)))

    return train, fewshot_ids, test


class MockArgs:
    """Mock args object with required attributes."""
    def __init__(self, country="United States", seed=27565976):
        self.country = country
        self.seed = seed


def test_load_data_determinism():
    """Test that load_data returns identical results across multiple calls."""
    args = MockArgs()

    # Call load_data multiple times
    results = []
    for i in range(3):
        train, fewshot_ids, test = load_data(args)
        results.append((train, fewshot_ids, test))

    # Compare all results
    for i in range(1, len(results)):
        train1, fewshot_ids1, test1 = results[0]
        train_i, fewshot_ids_i, test_i = results[i]

        # Check lengths
        assert len(train1) == len(train_i), f"Train lengths differ: {len(train1)} vs {len(train_i)}"
        assert len(fewshot_ids1) == len(fewshot_ids_i), f"Fewshot_ids lengths differ"
        assert len(test1) == len(test_i), f"Test lengths differ: {len(test1)} vs {len(test_i)}"

        # Check fewshot_ids match
        assert fewshot_ids1 == fewshot_ids_i, f"Fewshot_ids differ between call 0 and call {i}"

        # Check train items match (order and content)
        for j, (item1, item_i) in enumerate(zip(train1, train_i)):
            assert item1 == item_i, f"Train item {j} differs between call 0 and call {i}"

        # Check test items match (order and content)
        for j, (item1, item_i) in enumerate(zip(test1, test_i)):
            assert item1 == item_i, f"Test item {j} differs between call 0 and call {i}"

    print(f"SUCCESS: load_data returns identical outputs across {len(results)} calls")
    print(f"  - Train size: {len(results[0][0])}")
    print(f"  - Test size: {len(results[0][2])}")
    print(f"  - Fewshot IDs: {len(results[0][1])}")

    # Also test with different seeds to make sure seed affects output
    args2 = MockArgs(seed=12345)
    train_seed2, _, test_seed2 = load_data(args2)

    # The output should be different with a different seed
    train_original = results[0][0]
    if len(train_original) > 0 and len(train_seed2) > 0:
        # Check if at least the order is different
        differs = False
        for item1, item2 in zip(train_original, train_seed2):
            if item1.get('consistency_id') != item2.get('consistency_id'):
                differs = True
                break
        if differs:
            print("  - Verified: Different seeds produce different orderings")
        else:
            print("  - Note: Different seeds produced same ordering (may be expected for small datasets)")


def test_train_test_no_overlap():
    """Test that train and test sets have no overlapping items."""
    args = MockArgs()
    train, fewshot_ids, test = load_data(args)

    # Convert dicts to frozensets of tuples for hashable comparison
    def dict_to_hashable(d):
        return frozenset((k, str(v)) for k, v in d.items())

    train_set = {dict_to_hashable(item) for item in train}
    test_set = {dict_to_hashable(item) for item in test}

    overlap = train_set & test_set
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping items between train and test"

    # Also check by consistency_id to ensure no group leakage
    train_consistency_ids = {item['consistency_id'] for item in train}
    test_consistency_ids = {item['consistency_id'] for item in test}

    consistency_overlap = train_consistency_ids & test_consistency_ids
    assert len(consistency_overlap) == 0, (
        f"Found {len(consistency_overlap)} overlapping consistency_ids between train and test: {consistency_overlap}"
    )

    print(f"SUCCESS: No overlap between train and test")
    print(f"  - Train items: {len(train)}, unique consistency_ids: {len(train_consistency_ids)}")
    print(f"  - Test items: {len(test)}, unique consistency_ids: {len(test_consistency_ids)}")


if __name__ == "__main__":
    test_load_data_determinism()
    test_train_test_no_overlap()
