import asyncio
import json
import math
import random
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

from core.llm_api.llm import ModelAPI
from core.utils import setup_environment
from src.experiments.ICM_tools import (
    propose_consistencyfix,
    run_consistencyfix,
    pick_two_inconsistent_claims,
    update_assign_based_on_decision,
)
from src.model_querying.prompt_creation import (
    get_judge_prompt_fewshot,
    get_judge_prompt_zeroshot,
)
from src.model_querying.solution_extraction import (
    get_yes_no_diff_logprobs
)
from src.pipeline.pipeline import Pipeline, PipelineConfig
from src.tools.dataloaders import load_assignments
from src.tools.path_utils import get_root_directory


# Global model_api instance (initialized in main)
model_api = None


def calculate_accuracy(train_data, inconsistent_pairs):
    train_probs = []
    for i in train_data.values():
        if i["label"] is None:
            continue
        if i["label"] == 1:
            train_probs.append(i["score"])
        else:
            train_probs.append(-i["score"])
    if len(train_probs) == 0:
        train_prob = 0
    else:
        train_prob = np.mean(train_probs)

    return {
        "train_accuracy": 0
        if len(train_data) == 0
        else np.mean([i["label"] == i["vanilla_label"] for i in train_data.values()]),
        "train_prob": train_prob,
        "train_size": len(train_data),
        "inconsistent_num": len(inconsistent_pairs)
    }


def update_assign(data):
    for key, value in data.items():
        if value["score"] > 0:
            value["label"] = 1
        else:
            value["label"] = 0
    return data


async def fix_inconsistency(demonstrations, cur_metric, name, args, iter=0, K=20):
    """
    Fix inconsistencies in label assignments.

    Now async to support batched vLLM inference.
    """
    if cur_metric["inconsistent_num"] == 0:
        return demonstrations, cur_metric

    cur_pool = {k: v for k, v in demonstrations.items() if v["label"] is not None}
    assignment = cur_pool

    best_metric = cur_metric
    best_assignment = assignment
    best_decision_id = None

    for k in range(K):
        pipeline = propose_consistencyfix(
            name=name,
            iter=f"{iter}-{k}",
            assignment=assignment,
        )
        results = await pipeline.run()
        decisions = results["pick_two_inconsistent_claims"]
        assignment = results["get_assign"]

        for decision_id, decision in enumerate(decisions.values()):
            tmp_decision_metric_list = []
            tmp_decision_assignment_list = []

            for score_idx, score in enumerate([0, 1]):
                tmp_decision = deepcopy(decision)
                tmp_decision["score"] = score
                tmp_assignment = update_assign_based_on_decision(
                    deepcopy(assignment), tmp_decision
                )
                tmp_pipeline = run_consistencyfix(
                    model=args.model,
                    name=name,
                    iter=f"{iter}-{k}-{decision_id}-{score_idx}",
                    assignment=tmp_assignment,
                    model_api=model_api,
                )
                tmp_results = await tmp_pipeline.run()
                tmp_metric = tmp_results["evaluate"]
                tmp_decision_metric_list.append(tmp_metric)
                tmp_decision_assignment_list.append(tmp_assignment)

            tmp_best_decision_id = np.argmax(
                [get_energy(i, args.alpha) for i in tmp_decision_metric_list]
            )
            tmp_assignment = tmp_decision_assignment_list[tmp_best_decision_id]
            tmp_metric = tmp_decision_metric_list[tmp_best_decision_id]

            if get_energy(tmp_metric, args.alpha) >= get_energy(best_metric, args.alpha):
                best_decision_id = decision_id
                best_metric = tmp_metric
                best_assignment = tmp_assignment
                break

        if best_decision_id is None:
            break
        elif best_metric["inconsistent_num"] == 0:
            assignment = best_assignment
            break
        else:
            assignment = best_assignment
            best_decision_id = None

    for k in assignment:
        demonstrations[k] = assignment[k]

    return demonstrations, best_metric


def get_pipeline_batched(
    model,
    name=None,
    use_cache=True,
    num_problems=None,
    decision_id=None,
    iter=None,
    assignment=None,
):
    """
    Create a pipeline using batched vLLM inference.
    """
    pipeline_name = f"iterative-truth-assign-iter-{iter}-batched"
    if decision_id is not None:
        pipeline_name += f"-{decision_id}"
    if name is not None:
        pipeline_name += "-" + name

    pipeline_config = PipelineConfig(
        pipeline_name,
        openai_fraction_rate_limit=0.99,
        num_problems=num_problems,
        use_cache=use_cache,
    )
    pipeline = Pipeline(pipeline_config, model_api=model_api)

    assert assignment is not None
    initial_assign = pipeline.add_load_data_step(
        "get_assign", load_assignments, assignment
    )

    def add_train_demonstrations(train_data):
        copy_data = deepcopy(train_data)
        copy_data = {k: v for k, v in copy_data.items() if v["label"] is not None}
        keys = list(copy_data.keys())
        saved_keys = [
            "prompt",
            "question",
            "choice",
            "country",
            "consistency_id",
            "label",
            "vanilla_label",
        ]
        values = []
        for i in copy_data.values():
            values.append({saved_key: i[saved_key] for saved_key in saved_keys if saved_key in i})

        for idx, key in enumerate(keys):
            tmp_keys, tmp_values = [], []
            for j, (prev_key, prev_value) in enumerate(zip(keys, values)):
                if j != idx:
                    tmp_keys.append(prev_key)
                    tmp_values.append(prev_value)

            demos = {
                prev_key: prev_value
                for j, (prev_key, prev_value) in enumerate(zip(tmp_keys, tmp_values))
            }

            sorted_demos = {}
            for k, v in demos.items():
                q = v["consistency_id"]
                if q not in sorted_demos:
                    sorted_demos[q] = []
                sorted_demos[q].append((k, v))

            out_sorted_demos = {}
            for group in sorted_demos.values():
                for k, v in group:
                    out_sorted_demos[k] = v

            copy_data[key]["demonstration"] = out_sorted_demos

        return copy_data

    merged_train_data = pipeline.add_transformation_step(
        "add_train_demonstration",
        add_train_demonstrations,
        dependencies=[initial_assign],
    )

    # Use batched query step for vLLM
    get_train_preds = pipeline.add_batched_query_step(
        "get_train_preds",
        model,
        get_judge_prompt_fewshot,
        dependencies=[merged_train_data],
        logprobs=5,
        max_tokens=1,
        use_cache=use_cache,
    )

    pick_claims = pipeline.add_transformation_step(
        "pick_two_inconsistent_claims",
        pick_two_inconsistent_claims,
        dependencies=[initial_assign],
    )

    pipeline.add_eval_step(
        "evaluate",
        calculate_accuracy,
        dependencies=[get_train_preds, pick_claims],
    )
    return pipeline


async def predict_assignment(model, example, demonstrations):
    """
    Predict label for a single example using few-shot prompting.
    """
    demos = [
        v
        for k, v in demonstrations.items()
        if k != example["uid"] and v["label"] is not None
    ]

    prompt = get_judge_prompt_fewshot(
        example,
        demos,
        pipeline=False,
    )

    responses = await model_api(
        model,
        prompt,
        logprobs=5,
        max_tokens=1,
        temperature=0.0,
    )

    try:
        if hasattr(responses[0], 'logprobs'):
            logprobs = responses[0].logprobs
        else:
            logprobs = responses[0]["response"]["logprobs"]
        score = get_yes_no_diff_logprobs(logprobs)
    except Exception as e:
        print(f"Error in predict_assignment: {e}")
        score = 0

    new_label = score > 0
    return int(new_label)


async def predict_assignment_zero_shot(model, example, is_chat_model=False):
    """
    Predict label for a single example using zero-shot prompting.

    Args:
        model: The model to use for prediction
        example: The example to evaluate
        is_chat_model: If True, use instruction-style prompt for chat models
    """
    prompt = get_judge_prompt_zeroshot(example, pipeline=False, is_chat_model=is_chat_model)

    responses = await model_api(
        model,
        prompt,
        logprobs=5,
        max_tokens=1,
        temperature=0.0,
    )

    try:
        if hasattr(responses[0], 'logprobs'):
            logprobs = responses[0].logprobs
        else:
            logprobs = responses[0]["response"]["logprobs"]
        score = get_yes_no_diff_logprobs(logprobs)
    except Exception as e:
        print(f"Error in predict_assignment_zero_shot: {e}")
        score = 0

    new_label = score > 0
    return int(new_label)


def get_temperature(iteration, initial_temp, final_temp, decay_rate, schedule="log"):
    """Calculate the temperature for simulated annealing."""
    if schedule == "exp":
        return max(final_temp, initial_temp * (decay_rate**iteration))
    elif schedule == "log":
        return max(final_temp, initial_temp / (1 + 2.75 * np.log(1 + iteration)))
    else:
        assert False


def get_energy(metric, alpha):
    return alpha * metric["train_prob"] - metric["inconsistent_num"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=3.5)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B")
    parser.add_argument("--num_seed", type=int, default=12)
    parser.add_argument("--K", type=int, default=1000)
    parser.add_argument("--consistency_fix_K", type=int, default=20)
    parser.add_argument("--decay", type=float, default=0.995)
    parser.add_argument("--initial_T", type=float, default=10)
    parser.add_argument("--final_T", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default="log")

    # vLLM configuration
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--kv_cache_dtype", type=str, default="fp8")
    parser.add_argument("--enable_chunked_prefill", type=bool, default=False)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="Fraction of GPU memory to use (0.0-1.0)")
    parser.add_argument("--max_model_len", type=int, default=15000,
                        help="Maximum sequence length")
    parser.add_argument("--max_num_batched_tokens", type=int, default=115072,
                        help="Maximum number of batched tokens per iteration")
    parser.add_argument("--max_num_seqs", type=int, default=64,
                        help="Maximum number of sequences per iteration")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "binary"],
                        help="Dataset to use: 'all' for all questions, 'binary' for binary questions only")

    args = parser.parse_args()
    return args


def load_data(args, seed):
    if args.dataset == "binary":
        data_file = "transformed_global_opinions_binary.json"
    else:
        data_file = "transformed_global_opinions.json"
    with open(get_root_directory() / "data" / data_file) as f:
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
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    total_items = len(country_data)
    target_train_items = min(int(total_items * 0.75), 200)
    target_test_items = min(76, int(total_items * .25))    

    train_ids = []
    train_group_ids = []
    for gid in group_ids:
        if len(train_ids) >= target_train_items:
            break
        train_ids.extend(consistency_groups[gid])
        train_group_ids.append(gid)
    train_ids = train_ids[:target_train_items]

    # Remaining groups go to test
    test_group_ids = [gid for gid in group_ids if gid not in train_group_ids]
    test_ids = []
    for gid in test_group_ids:
        if len(test_ids) >= target_test_items:
            break
        test_ids.extend(consistency_groups[gid])
    test_ids = test_ids[:target_test_items]

    # Build train and test lists
    train = [country_data[i] for i in train_ids]
    test = [country_data[i] for i in test_ids]

    # fewshot_ids is just all indices into train
    fewshot_ids = list(range(len(train)))

    return train, fewshot_ids, test


def initialize(train, fewshot_ids, args):
    demonstrations = {}
    whole_ids = []
    args.num_seed = len(train)
    random_init_labels = [1] * (args.num_seed // 2) + [0] * (args.num_seed // 2)
    random.shuffle(random_init_labels)

    for id, i in enumerate(fewshot_ids):
        item = train[i]
        item["vanilla_label"] = item["label"]
        item["uid"] = id
        whole_ids.append(item["uid"])
        if id >= args.num_seed:
            item["label"] = 0
            item["type"] = "predict"
        else:
            item["type"] = "seed"
            item["label"] = random_init_labels[id]
        demonstrations[id] = item

    return demonstrations, whole_ids


async def golden_supervision_main(args, train, fewshot_ids, test, icm_demonstrations):
    """
    Benchmark using golden (ground truth) labels for demonstrations.

    Samples a subset of examples to avoid context overload which can
    degrade in-context learning performance.

    Args:
        icm_demonstrations: Dict of ICM demonstrations (to ensure we sample from same pool)
        num_examples: Number of examples to sample for demonstrations
    """
    print("\nRunning golden supervision benchmark...")

    # Build demonstrations only for UIDs that ICM labeled (to ensure fair comparison)
    all_demonstrations = {}
    for id, i in enumerate(fewshot_ids):
        if id in icm_demonstrations:
            item = train[i]
            item["uid"] = id
            all_demonstrations[id] = item
        else:
            assert False

    # DEBUG: Verify key ordering matches icm_demonstrations
    assert list(all_demonstrations.keys()) == list(icm_demonstrations.keys()), \
        f"Key mismatch: all_demonstrations keys differ from icm_demonstrations keys"
    
    max_uid = max(all_demonstrations.keys())
    test_acc_list = []
    for i in range(10):
        # Randomize demonstration order with fixed seed for reproducibility
        demo_rng = random.Random()
        shuffled_uids = list(all_demonstrations.keys())
        demo_rng.shuffle(shuffled_uids)

        demonstrations = {uid: all_demonstrations[uid] for uid in shuffled_uids}

        correct_cnt = 0

        for idx, item in enumerate(tqdm(test, desc="Golden supervision evaluation")):
            item['uid'] = max_uid + 1 + idx
            new_label = await predict_assignment(args.model, item, demonstrations)
            if item['label'] == new_label:
                correct_cnt += 1

        test_accuracy = correct_cnt / len(test)
        test_acc_list.append(test_accuracy)

    mean_test_acc = np.mean(test_acc_list)
    std_dev_test_acc = np.std(test_acc_list)

    print(f"Golden Supervision Test Accuracy: {mean_test_acc:.4f}")

    return mean_test_acc, std_dev_test_acc


async def zero_shot_chat_main(args, test):
    """
    Benchmark using zero-shot prompting with chat/instruct model.
    """
    print("\nRunning zero-shot chat benchmark...")

    # Determine instruct model based on base model
    model_mapping = {
        'meta-llama/Llama-3.1-8B': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        'meta-llama/Llama-3.1-70B': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        'meta-llama/Llama-3.1-405B': 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
        'RedHatAI/Meta-Llama-3.1-8B-FP8': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        'RedHatAI/Meta-Llama-3.1-70B-FP8': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    }
    instruct_model = model_mapping.get(args.model, args.model + '-Instruct-Turbo')
    print(f"In zero-shot chat method: Using {instruct_model}")
    test_acc_list = []
    for i in range(5):
        correct_cnt = 0
        for idx, item in enumerate(tqdm(test, desc="Zero-shot chat evaluation")):
            new_label = await predict_assignment_zero_shot(instruct_model, item, is_chat_model=True)
            if item['label'] == new_label:
                correct_cnt += 1

        test_accuracy = correct_cnt / len(test)
        test_acc_list.append(test_accuracy)
    mean_test_acc = np.mean(test_acc_list)
    std_dev_test_acc = np.std(test_acc_list)

    print(f"Zero-shot Chat Test Accuracy: {mean_test_acc:.4f}")

    return mean_test_acc, std_dev_test_acc


async def zero_shot_pretrained_main(args, test):
    """
    Benchmark using zero-shot prompting with pretrained (base) model.
    """
    print("\nRunning zero-shot pretrained benchmark...")

    correct_cnt = 0
    print(f"In zero-shot base method: Using {args.model}")
    print_prompt_ind = False
    for idx, item in enumerate(tqdm(test, desc="Zero-shot pretrained evaluation")):
        if idx == len(test)-1:
            print_prompt_ind=True
        new_label = await predict_assignment_zero_shot(args.model, item, is_chat_model=False, print_prompt=print_prompt_ind)
        if item['label'] == new_label:
            correct_cnt += 1

    test_accuracy = correct_cnt / len(test)
    print(f"Zero-shot Pretrained Test Accuracy: {test_accuracy:.4f}")

    return test_accuracy


async def compare_labels_by_num_examples(args, train, fewshot_ids, test, icm_demonstrations, seed):
    """
    Compare test accuracy as a function of number of in-context examples
    for gold labels, ICM labels, and random labels (accuracy-matched).

    The number of ICM-labeled examples dictates the threshold for comparison.

    Args:
        args: Command line arguments
        train: Training data with gold labels
        fewshot_ids: Indices into train for few-shot examples
        test: Test data
        icm_demonstrations: Dict of demonstrations with ICM-assigned labels (None labels already purged)
        seed: Random seed for reproducibility

    Returns:
        Dict with num_examples, gold_acc, icm_acc, random_acc, icm_train_acc lists
    """
    print("\n" + "="*50)
    print("Comparing Labels by Number of Examples")
    print("="*50)

    # Base all_uids on ICM demonstrations (which dictates the threshold)
    all_uids = list(icm_demonstrations.keys())
    print(f"ICM provided labels for {len(all_uids)} examples")

    # Build gold demonstrations only for UIDs that exist in icm_demonstrations
    gold_demonstrations = {}
    for id, i in enumerate(fewshot_ids):
        if id in icm_demonstrations:
            item = train[i].copy()
            item["uid"] = id
            gold_demonstrations[id] = item
        else:
            assert False

    # DEBUG: Verify labels match between gold_demonstrations and icm_demonstrations vanilla_labels
    for uid in all_uids:
        gold_label = gold_demonstrations[uid]['label']
        icm_vanilla = icm_demonstrations[uid].get('vanilla_label', icm_demonstrations[uid]['label'])
        if gold_label != icm_vanilla:
            print(f"[DEBUG] Label mismatch at UID {uid}: gold={gold_label}, icm_vanilla={icm_vanilla}")
    print(f"[DEBUG compare_labels] Verified labels for {len(all_uids)} demonstrations")
    
    # Group UIDs by consistency_id for sampling
    consistency_groups = {}
    for uid in all_uids:
        cid = icm_demonstrations[uid]['consistency_id']
        if cid not in consistency_groups:
            consistency_groups[cid] = []
        consistency_groups[cid].append(uid)

    # Number of examples to test
    num_examples_list = [10, 20, 50, 75, 100, 150, 200, 300, 400, 500]
    # Filter to only include values <= available ICM examples
    num_examples_list = [n for n in num_examples_list if n <= len(all_uids)]
    if len(all_uids) not in num_examples_list:
        num_examples_list += [len(all_uids)]

    results = {
        'num_examples': [],
        'gold_acc': [],
        'icm_acc': [],
        'random_acc': [],
        'icm_train_acc': [],
    }

    rng = random.Random()
    for i in range(10):
        for num_examples in tqdm(num_examples_list, desc="Comparing labels by num examples"):
            # Sample by consistency groups: add entire groups until num_examples is reached
            group_ids = list(consistency_groups.keys())
            rng.shuffle(group_ids)

            sampled_uids = []
            for gid in group_ids:
                group_uids = consistency_groups[gid]
                remaining = num_examples - len(sampled_uids)
                if remaining <= 0:
                    break
                if len(group_uids) <= remaining:
                    # Add entire group
                    sampled_uids.extend(group_uids)
                else:
                    # Add partial group to reach exactly num_examples
                    sampled_uids.extend(group_uids[:remaining])

            # 1. Gold labels test accuracy
            gold_demos_subset = {uid: gold_demonstrations[uid].copy() for uid in sampled_uids}
            max_uid = max(gold_demos_subset.keys())

            gold_correct = 0
            gold_predictions = []  # DEBUG: Store predictions for comparison
            test_items = []
            for idx, item in enumerate(test):
                item_copy = item.copy()
                test_items.append(item_copy)
                item_copy['uid'] = max_uid + 1 + idx
                new_label = await predict_assignment(args.model, item_copy, gold_demos_subset)
                gold_predictions.append(new_label)  # DEBUG
                if item['label'] == new_label:
                    gold_correct += 1
            gold_acc = gold_correct / len(test)
            print(f"  Gold labels test accuracy: {gold_acc:.4f}")

            # 3. Compute ICM training accuracy (how well ICM labels match gold labels)
            icm_train_matches = 0
            for uid in sampled_uids:
                gold_label = gold_demonstrations[uid]['label']
                icm_label = icm_demonstrations[uid]['label']
                if gold_label == icm_label:
                    icm_train_matches += 1
            icm_train_acc = icm_train_matches / len(sampled_uids)

            # 4. Generate random labels with same accuracy as ICM
            # Start with gold labels, then flip some to match ICM's accuracy
            random_demos_subset = {uid: gold_demonstrations[uid].copy() for uid in sampled_uids}

            # Calculate how many labels to flip to achieve same accuracy as ICM
            num_correct_needed = int(len(sampled_uids) * icm_train_acc)
            num_to_flip = len(sampled_uids) - num_correct_needed

            # Randomly select which labels to flip
            if num_to_flip > 0:
                uids_to_flip = rng.sample(sampled_uids, min(num_to_flip, len(sampled_uids)))
                for uid in uids_to_flip:
                    random_demos_subset[uid]['label'] = 1 - random_demos_subset[uid]['label']

            #check that random training accuracy is +/- 1 of ICM training accuracy
            random_acc_train_matches = 0
            for uid in sampled_uids:
                gold_label = gold_demonstrations[uid]['label']
                r_label = random_demos_subset[uid]['label']
                if gold_label == r_label:
                    random_acc_train_matches += 1
            assert random_acc_train_matches <= icm_train_matches

            if model_api._vllm_client and model_api._vllm_client._engine:
                    model_api._vllm_client._engine.llm_engine.reset_prefix_cache()

            random_correct = 0
            for idx, item in enumerate(test):
                item_copy = item.copy()
                item_copy['uid'] = max_uid + 1 + idx
                new_label = await predict_assignment(args.model, item_copy, random_demos_subset)
                if item['label'] == new_label:
                    random_correct += 1
            random_acc = random_correct / len(test)
            print(f"  Random labels test accuracy: {random_acc:.4f}")

            # Store results
            results['num_examples'].append(num_examples)
            results['gold_acc'].append(gold_acc)
            results['random_acc'].append(random_acc)
            results['icm_train_acc'].append(icm_train_acc)

    # Convert to DataFrame, group by num_examples, compute mean and std
    df = pd.DataFrame(results)
    grouped = df.groupby('num_examples').agg(['mean', 'std'])

    # Build aggregated results dict
    aggregated_results = {
        'num_examples': grouped.index.tolist(),
        'gold_acc': grouped[('gold_acc', 'mean')].tolist(),
        'icm_acc': [],  # Empty since not computed in loop
        'random_acc': grouped[('random_acc', 'mean')].tolist(),
        'icm_train_acc': grouped[('icm_train_acc', 'mean')].tolist(),
        'gold_acc_std': grouped[('gold_acc', 'std')].tolist(),
        'icm_acc_std': [],  # Empty since not computed in loop
        'random_acc_std': grouped[('random_acc', 'std')].tolist(),
        'icm_train_acc_std': grouped[('icm_train_acc', 'std')].tolist(),
    }

    return aggregated_results


async def async_main(args, seed, country):
    """Main async entry point."""
    global model_api
    # Set country on args so downstream functions can access it
    args.country = country

    setup_environment(logger_level="error", openai_tag='TOGETHER_API_KEY')
    random_seed = seed
    # Load data
    train, fewshot_ids, test = load_data(args, random_seed)
    test_size = len(test)

    print(f'Model: {args.model}, Country: {args.country}, Train size: {len(train)}, Test size: {len(test)}')

    demonstrations, _ = initialize(train, fewshot_ids, args)
    icm_demos = demonstrations
    # Run golden supervision benchmark
    print("\n" + "="*50)
    print(f"Running Golden Supervision Benchmark for {country}")
    print("="*50)
    train, fewshot_ids, test = load_data(args, random_seed)
    golden_acc_mean, gold_acc_std_dev = await golden_supervision_main(args, train, fewshot_ids, test, icm_demos)

    # Run zero-shot chat benchmark
    print("\n" + "="*50)
    print(f"Running Zero-shot Chat Benchmark for {country}")
    print("="*50)
    _, _, test = load_data(args, random_seed)  # Reload test
    chat_acc_mean, chat_acc_std_dev = await zero_shot_chat_main(args, test)

    # Run zero-shot pretrained benchmark
    print("\n" + "="*50)
    print(f"Running Zero-shot Pretrained Benchmark for {country}")
    print("="*50)
    # Clear prefix cache before compare_labels
    if model_api._vllm_client and model_api._vllm_client._engine:
        model_api._vllm_client._engine.llm_engine.reset_prefix_cache()
    _, _, test = load_data(args, random_seed)  # Reload test
    pretrained_acc = await zero_shot_pretrained_main(args, test)

    # Clear prefix cache before compare_labels
    if model_api._vllm_client and model_api._vllm_client._engine:
        model_api._vllm_client._engine.llm_engine.reset_prefix_cache()
    
    # Compare labels by number of examples (gold vs ICM vs random)
    train, fewshot_ids, test = load_data(args, random_seed)  # Reload data
    icm_demos = demonstrations
    comparison_results = await compare_labels_by_num_examples(
        args, train, fewshot_ids, test, icm_demos, random_seed
    )
    
    print("\n" + "="*50)
    print(f"RESULTS SUMMARY -- {country}")
    print("="*50)
    print(f"Golden Supervision Mean Test Accuracy:      {golden_acc_mean*100:.2f}%")
    print(f"Zero-shot (Chat) Mean Test Accuracy:        {chat_acc_mean*100:.2f}%")
    print(f"Zero-shot (Pretrained) Test Accuracy:  {pretrained_acc*100:.2f}%")
    print(f"Comparison results: {comparison_results}")
 
    return {
        "country": country,
        "golden": golden_acc_mean,
        "golden_std_dev": gold_acc_std_dev,
        "chat": chat_acc_mean,
        "chat_std_dev": chat_acc_std_dev,
        "pretrained": pretrained_acc,
        "pretrained_std_dev": 0,
        "test_size": test_size,
        "comparison": comparison_results,
    }


if __name__ == "__main__":
    args = get_args()

    #countries = ["France", "Germany", "Japan", "Russia", "United States"]
    countries = ["United States", "France", "Japan"]

    # Initialize ModelAPI with vLLM configuration
    model_api = ModelAPI(
        openai_fraction_rate_limit=0.99,
        use_vllm=True,
        vllm_model_name=args.model,
        vllm_tensor_parallel_size=args.tensor_parallel_size,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_max_model_len=args.max_model_len,
        vllm_max_num_batched_tokens=args.max_num_batched_tokens,
        vllm_max_num_seqs=args.max_num_seqs,
        vllm_enable_prefix_caching=True,
        vllm_enable_chunked_prefill=args.enable_chunked_prefill,
        vllm_kv_cache_dtype=args.kv_cache_dtype
    )

    # Use fixed seed
    seed = 733244742

    try:
        # Collect results for each country
        all_results = {}
        for country in countries:
            print(f"\n{'#'*60}")
            print(f"# Processing country: {country}")
            print(f"{'#'*60}")
            all_results[country] = asyncio.run(async_main(args, seed, country))

        # Aggregate results weighted by test set size
        total_test_size = sum(r["test_size"] for r in all_results.values())

        aggregated = {"country": "aggregated"}
        for benchmark in ["golden", "chat", "pretrained"]:
            weighted_acc = sum(
                r[benchmark] * r["test_size"] for r in all_results.values()
            ) / total_test_size

            weighted_var = sum(
                r[benchmark + '_std_dev']**2 * (r["test_size"]/total_test_size)**2 for r in all_results.values()
            )
            aggregated[benchmark] = weighted_acc
            aggregated[benchmark + '_std_dev'] = np.sqrt(weighted_var)
        aggregated['comparison'] = None
        
        # Print aggregated summary
        print("\n" + "="*60)
        print("AGGREGATED RESULTS (weighted by test set size)")
        print("="*60)
        print(f"Total test examples: {total_test_size}")
        print(f"Golden Supervision:      {aggregated['golden']*100:.2f}%")
        print(f"Zero-shot (Chat):        {aggregated['chat']*100:.2f}%")
        print(f"Zero-shot (Pretrained):  {aggregated['pretrained']*100:.2f}%")

        # Aggregate comparison results across countries (weighted by test set size)
        # Group by num_examples and compute weighted averages
        comparison_by_num_examples = {}
        for country, results in all_results.items():
            test_size = results["test_size"]
            comparison = results["comparison"]
            for i, num_ex in enumerate(comparison['num_examples']):
                if num_ex not in comparison_by_num_examples:
                    comparison_by_num_examples[num_ex] = {
                        'gold_acc_weighted': 0.0,
                        'random_acc_weighted': 0.0,
                        'icm_train_acc_weighted': 0.0,
                        'total_test_size': 0,
                        'gold_acc_std_weighted': 0,
                        'random_acc_std_weighted': 0,
                        'icm_train_acc_std_weighted': 0
                    }
                comparison_by_num_examples[num_ex]['gold_acc_weighted'] += comparison['gold_acc'][i] * test_size
                comparison_by_num_examples[num_ex]['random_acc_weighted'] += comparison['random_acc'][i] * test_size
                comparison_by_num_examples[num_ex]['icm_train_acc_weighted'] += comparison['icm_train_acc'][i] * test_size
                comparison_by_num_examples[num_ex]['gold_acc_std_weighted'] += (comparison['gold_acc_std'][i])**2 * (test_size**2)
                comparison_by_num_examples[num_ex]['random_acc_std_weighted'] += (comparison['random_acc_std'][i])**2 * (test_size**2)
                comparison_by_num_examples[num_ex]['icm_train_acc_std_weighted'] += (comparison['icm_train_acc_std'][i])**2 * (test_size**2)
                comparison_by_num_examples[num_ex]['total_test_size'] += test_size

        # Build aggregated comparison results with weighted averages
        aggregated_comparison = {
            'num_examples': sorted(comparison_by_num_examples.keys()),
            'gold_acc': [],
            'icm_acc': [],
            'random_acc': [],
            'icm_train_acc': [],
            'gold_acc_std': [],
            'icm_acc_std': [],
            'random_acc_std': [],
            'icm_train_acc_std': []
        }
        for num_ex in aggregated_comparison['num_examples']:
            data = comparison_by_num_examples[num_ex]
            total = data['total_test_size']
            aggregated_comparison['gold_acc'].append(data['gold_acc_weighted'] / total)
            aggregated_comparison['gold_acc_std'].append(data['gold_acc_std_weighted'] / (total**2))
            aggregated_comparison['random_acc'].append(data['random_acc_weighted'] / total)
            aggregated_comparison['random_acc_std'].append(data['random_acc_std_weighted'] / (total**2))
            aggregated_comparison['icm_train_acc'].append(data['icm_train_acc_weighted'] / total)
            aggregated_comparison['icm_train_acc_std'].append(data['icm_train_acc_std_weighted'] / (total**2))

        print(aggregated_comparison)

        aggregated['comparison'] = aggregated_comparison

        # Print JSON-friendly outputs
        print("\n" + "="*60)
        print("JSON OUTPUTS")
        print("=\n"*60)
        print("[")
        for country, results in all_results.items():
            print(json.dumps(results, indent=2) + ",")
        print(json.dumps(aggregated, indent=2))
        print("]")

    finally:
        # Gracefully shutdown vLLM engine
        model_api.shutdown()
