import asyncio
import json
import math
import random
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import argparse
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
        return max(final_temp, initial_temp / (1 + 1.35 * np.log(1 + iteration)))
    else:
        assert False


def get_energy(metric, alpha):
    return alpha * metric["train_prob"] - metric["inconsistent_num"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=5)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B")
    parser.add_argument("--num_seed", type=int, default=8)
    parser.add_argument("--K", type=int, default=1500)
    parser.add_argument("--consistency_fix_K", type=int, default=20)
    parser.add_argument("--decay", type=float, default=0.995)
    parser.add_argument("--initial_T", type=float, default=10)
    parser.add_argument("--final_T", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default="log")

    # vLLM configuration
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="Fraction of GPU memory to use (0.0-1.0)")
    parser.add_argument("--max_model_len", type=int, default=20000,
                        help="Maximum sequence length")
    parser.add_argument("--max_num_batched_tokens", type=int, default=196608,
                        help="Maximum number of batched tokens per iteration")
    parser.add_argument("--max_num_seqs", type=int, default=485,
                        help="Maximum number of sequences per iteration")

    args = parser.parse_args()
    return args


def load_data(args, seed):
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
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    total_items = len(country_data)
    target_train_items = min(int(total_items * 0.75), 300)
    target_test_items = min(100, int(total_items * .25))    

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
        if len(test_ids) >= target_test_items:
            break
        test_ids.extend(consistency_groups[gid])

    # Build train and test lists
    train = [country_data[i] for i in train_ids]
    test = [country_data[i] for i in test_ids]

    # fewshot_ids is just all indices into train
    fewshot_ids = list(range(len(train)))

    return train, fewshot_ids, test


def initialize(train, fewshot_ids, args):
    demonstrations = {}
    whole_ids = []

    random_init_labels = [1] * (args.num_seed // 2) + [0] * (args.num_seed // 2)
    random.shuffle(random_init_labels)

    for id, i in enumerate(fewshot_ids):
        item = train[i]
        item["vanilla_label"] = item["label"]
        item["uid"] = id
        whole_ids.append(item["uid"])
        if id >= args.num_seed:
            item["label"] = None
            item["type"] = "predict"
        else:
            item["type"] = "seed"
            item["label"] = random_init_labels[id]
        demonstrations[id] = item

    return demonstrations, whole_ids


async def icm_main(args, train, fewshot_ids, test):
    """
    Main ICM algorithm with test evaluation.

    Returns:
        Tuple of (test_accuracy, label_assignments, demonstrations)
    """
    train_size = len(fewshot_ids)
    demonstrations, whole_ids = initialize(train, fewshot_ids, args)

    cur_metric = {
        "train_prob": -1e6,
        "inconsistent_num": 100000,
        "train_accuracy": 1.0,
        "train_predict_distribution": {"0": 0, "1": 0},
        "train_label_distribution": {"0": 0, "1": 0},
    }

    print('init random labels = ', Counter([i['label'] for i in demonstrations.values() if i['type'] == 'seed']),
          'init label acc = ', np.mean([i['label'] == i['vanilla_label'] for i in demonstrations.values() if i['type'] == 'seed']))

    name = f"{args.country}-K{args.K}-initialsize{args.num_seed}-weighted{args.alpha}-decay{args.decay}-initialT{args.initial_T}-finalT{args.final_T}-scheduler{args.scheduler}"

    iter_count = 0
    flip_cnt = 0
    reject_cnt = 0
    new_label_sample = 0 #flip_cnt + reject_cnt = new_label_sample
    num_iterations = min(args.K, train_size * 4)
    for _ in tqdm(range(num_iterations), desc="ICM searching"):
        cur_pool = {k: v for k, v in demonstrations.items() if v["label"] is not None}

        if iter_count == 0:
            pipeline = get_pipeline_batched(
                args.model,
                name=name,
                num_problems=None,
                iter=iter_count,
                assignment=cur_pool,
            )
            results = await pipeline.run()
            cur_metric = results["evaluate"]

            demonstrations, cur_metric = await fix_inconsistency(
                demonstrations, cur_metric, name, args, iter=iter_count, K=args.consistency_fix_K
            )

        cur_pool = {k: v for k, v in demonstrations.items() if v["label"] is not None}

        # Weighted sampling - prioritize items in consistency groups that have some labeled items
        candidates_ids = whole_ids
        weights = [1 for _ in range(len(candidates_ids))]
        for i in candidates_ids:
            if i in cur_pool:
                same_consistency_group_ids = [j for j in candidates_ids if demonstrations[j]["consistency_id"] == demonstrations[i]["consistency_id"]]
                for j in same_consistency_group_ids:
                    if j not in cur_pool:
                        weights[j] = 100

        example_id = random.choices(candidates_ids, k=1, weights=weights)[0]

        new_label = await predict_assignment(
            args.model,
            demonstrations[example_id],
            cur_pool,
        )

        if demonstrations[example_id]["label"] != new_label:
            new_label_sample += 1
            tmp_demonstrations = deepcopy(demonstrations)
            tmp_demonstrations[example_id]["label"] = new_label

            dummy_metric = {
                "train_prob": -1e6,
                "inconsistent_num": 100000,
                "train_accuracy": 1.0,
                "train_predict_distribution": {"0": 0, "1": 0},
                "train_label_distribution": {"0": 0, "1": 0},
            }

            tmp_demonstrations, _ = await fix_inconsistency(
                tmp_demonstrations,
                dummy_metric,
                name + "newlabelexplore",
                args,
                iter=iter_count,
                K=10,
            )

            tmp_pool = {k: v for k, v in tmp_demonstrations.items() if v["label"] is not None}

            pipeline = get_pipeline_batched(
                model=args.model,
                name=name,
                num_problems=None,
                iter=iter_count,
                assignment=tmp_pool,
            )
            results = await pipeline.run()
            metric = results["evaluate"]

            T = get_temperature(
                flip_cnt, args.initial_T, args.final_T, args.decay, schedule=args.scheduler
            )

            if iter_count % 10 == 0:
                print(f"iter = {iter_count}, pool size = {len(cur_pool)}, cur acc = {cur_metric['train_accuracy']:.4f}, "
                      f"new acc = {metric['train_accuracy']:.4f}, cur score = {get_energy(cur_metric, args.alpha):.4f}, "
                      f"new score = {get_energy(metric, args.alpha):.4f}, cur inconsistent = {cur_metric['inconsistent_num']}, "
                      f"new inconsistent = {metric['inconsistent_num']}")

            accept_prob = math.exp((get_energy(metric, args.alpha) - get_energy(cur_metric, args.alpha)) / T)
            if random.random() < accept_prob:
                demonstrations = tmp_demonstrations
                flip_cnt += 1
                cur_metric = metric
            else:
                reject_cnt+=1

        iter_count += 1

    # Evaluate on test set
    print("\nEvaluating ICM on test set...")
    max_uid = max(demonstrations.keys())
    correct_cnt = 0
    label_assignments = {}

    for idx, item in enumerate(tqdm(test, desc="ICM test evaluation")):
        item['uid'] = max_uid + 1 + idx
        new_label = await predict_assignment(args.model, item, demonstrations)
        label_assignments[idx] = new_label
        item['new_label'] = new_label
        if item['label'] == new_label:
            correct_cnt += 1

    test_accuracy = correct_cnt / len(test)
    print(f"ICM Test Accuracy: {test_accuracy:.4f}")

    # Purge demonstrations where label is None before returning
    demonstrations = {k: v for k, v in demonstrations.items() if v['label'] is not None}

    return test_accuracy, label_assignments, reject_cnt, new_label_sample, demonstrations


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
    
    demonstrations = all_demonstrations

    max_uid = max(all_demonstrations.keys())
    correct_cnt = 0
    label_assignments = {}

    for idx, item in enumerate(tqdm(test, desc="Golden supervision evaluation")):
        item['uid'] = max_uid + 1 + idx
        new_label = await predict_assignment(args.model, item, demonstrations)
        label_assignments[idx] = new_label
        item['new_label'] = new_label
        if item['label'] == new_label:
            correct_cnt += 1

    test_accuracy = correct_cnt / len(test)
    print(f"Golden Supervision Test Accuracy: {test_accuracy:.4f}")

    return test_accuracy, label_assignments


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
    }
    instruct_model = model_mapping.get(args.model, args.model + '-Instruct-Turbo')

    correct_cnt = 0
    label_assignments = {}

    for idx, item in enumerate(tqdm(test, desc="Zero-shot chat evaluation")):
        new_label = await predict_assignment_zero_shot(instruct_model, item, is_chat_model=True)
        label_assignments[idx] = new_label
        item['new_label'] = new_label
        if item['label'] == new_label:
            correct_cnt += 1

    test_accuracy = correct_cnt / len(test)
    print(f"Zero-shot Chat Test Accuracy: {test_accuracy:.4f}")

    return test_accuracy, label_assignments


async def zero_shot_pretrained_main(args, test):
    """
    Benchmark using zero-shot prompting with pretrained (base) model.
    """
    print("\nRunning zero-shot pretrained benchmark...")

    correct_cnt = 0
    label_assignments = {}

    for idx, item in enumerate(tqdm(test, desc="Zero-shot pretrained evaluation")):
        new_label = await predict_assignment_zero_shot(args.model, item, is_chat_model=False)
        label_assignments[idx] = new_label
        item['new_label'] = new_label
        if item['label'] == new_label:
            correct_cnt += 1

    test_accuracy = correct_cnt / len(test)
    print(f"Zero-shot Pretrained Test Accuracy: {test_accuracy:.4f}")

    return test_accuracy, label_assignments


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

    results = {
        'num_examples': [],
        'gold_acc': [],
        'icm_acc': [],
        'random_acc': [],
        'icm_train_acc': [],
    }

    rng = random.Random(seed)

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
        for idx, item in enumerate(test):
            item_copy = item.copy()
            item_copy['uid'] = max_uid + 1 + idx
            new_label = await predict_assignment(args.model, item_copy, gold_demos_subset)
            if item['label'] == new_label:
                gold_correct += 1
        gold_acc = gold_correct / len(test)
        print(f"  Gold labels test accuracy: {gold_acc:.4f}")

        # 2. ICM labels test accuracy (same examples)
        icm_demos_subset = {}
        for uid in sampled_uids:
            icm_demos_subset[uid] = icm_demonstrations[uid].copy()

        icm_correct = 0
        for idx, item in enumerate(test):
            item_copy = item.copy()
            item_copy['uid'] = max_uid + 1 + idx
            new_label = await predict_assignment(args.model, item_copy, icm_demos_subset)
            if item['label'] == new_label:
                icm_correct += 1
        icm_acc = icm_correct / len(test)
        print(f"  ICM labels test accuracy: {icm_acc:.4f}")

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
        results['icm_acc'].append(icm_acc)
        results['random_acc'].append(random_acc)
        results['icm_train_acc'].append(icm_train_acc)

    return results


def plot_accuracy_vs_num_examples(results, country):
    """
    Plot test accuracy as a function of number of in-context examples.

    Args:
        results: Dict with num_examples, gold_acc, icm_acc, random_acc lists
        country: Country name for title
    """
    save_path = f"figure_2_accuracy_vs_examples_{country}.png"

    fig, ax = plt.subplots(figsize=(10, 6))

    num_examples = results['num_examples']

    ax.plot(num_examples, [acc * 100 for acc in results['gold_acc']],
            'o-', color='#FFD700', linewidth=2, markersize=8, label='Gold Labels')
    ax.plot(num_examples, [acc * 100 for acc in results['icm_acc']],
            's-', color='#58b6c0', linewidth=2, markersize=8, label='ICM Labels')
    ax.plot(num_examples, [acc * 100 for acc in results['random_acc']],
            '^-', color='#9658ca', linewidth=2, markersize=8, label='Random Labels (accuracy-matched)')

    ax.set_xlabel("Number of In-Context Examples", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(f"Test Accuracy vs Number of Examples - {country}", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y-axis limits with some padding
    all_accs = results['gold_acc'] + results['icm_acc'] + results['random_acc']
    min_acc = min(all_accs) * 100
    max_acc = max(all_accs) * 100
    padding = (max_acc - min_acc) * 0.1
    ax.set_ylim(max(0, min_acc - padding), min(100, max_acc + padding))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_test_accuracies(icm_acc, golden_acc, chat_acc, pretrained_acc, country):
    """
    Plot comparison of test accuracies across methods.
    """
    save_path = f"figure_1_persona_{country}.png"
    accuracies = [
        pretrained_acc * 100,
        chat_acc * 100,
        icm_acc * 100,
        golden_acc * 100,
    ]
    labels = [
        "Zero-shot (Pretrained)",
        "Zero-shot (Chat)",
        "ICM (Unsupervised)",
        "Golden Supervision",
    ]

    bar_colors = [
        "#9658ca",  # darker purple for zero-shot pretrained
        "#B366CC",  # purple for zero-shot chat
        "#58b6c0",  # teal for ICM
        "#FFD700",  # gold for golden supervision
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        x,
        accuracies,
        color=bar_colors,
        tick_label=labels,
        edgecolor="k",
        zorder=2
    )

    # Add hatching to zero-shot chat bar
    bars[1].set_hatch('...')

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(f"Test Accuracy Comparison - {country}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.grid(axis='y', zorder=1, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


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

    # Run ICM
    print("\n" + "="*50)
    print("Running ICM Algorithm")
    print("="*50)
    icm_acc, icm_labels, reject_cnt, new_label_sample, icm_demos = await icm_main(args, train, fewshot_ids, test)

    # Run golden supervision benchmark
    print("\n" + "="*50)
    print("Running Golden Supervision Benchmark")
    print("="*50)
    # Reload train data to get fresh labels
    train, fewshot_ids, test = load_data(args, random_seed)
    golden_acc, golden_labels = await golden_supervision_main(args, train, fewshot_ids, test, icm_demos)

    # Run zero-shot chat benchmark
    print("\n" + "="*50)
    print("Running Zero-shot Chat Benchmark")
    print("="*50)
    _, _, test = load_data(args, random_seed)  # Reload test
    chat_acc, chat_labels = await zero_shot_chat_main(args, test)

    # Run zero-shot pretrained benchmark
    print("\n" + "="*50)
    print("Running Zero-shot Pretrained Benchmark")
    print("="*50)
    _, _, test = load_data(args, random_seed)  # Reload test
    pretrained_acc, pretrained_labels = await zero_shot_pretrained_main(args, test)

    # Compare labels by number of examples (gold vs ICM vs random)
    train, fewshot_ids, test = load_data(args, random_seed)  # Reload data
    comparison_results = await compare_labels_by_num_examples(
        args, train, fewshot_ids, test, icm_demos, random_seed
    )
    plot_accuracy_vs_num_examples(comparison_results, args.country)

    # Print summary
    print("\n" + "="*50)
    print(f"RESULTS SUMMARY -- {args.country}")
    print("="*50)
    print(f"ICM new labels sampled {new_label_sample} times out of {args.K}")
    print(f"ICM reject cnt: {reject_cnt}")
    print(f"ICM (Unsupervised):      {icm_acc*100:.2f}%")
    print(f"Golden Supervision:      {golden_acc*100:.2f}%")
    print(f"Zero-shot (Chat):        {chat_acc*100:.2f}%")
    print(f"Zero-shot (Pretrained):  {pretrained_acc*100:.2f}%")

    # Plot results
    plot_test_accuracies(icm_acc, golden_acc, chat_acc, pretrained_acc, args.country)

    return {
        "icm": (icm_acc, icm_labels),
        "golden": (golden_acc, golden_labels),
        "chat": (chat_acc, chat_labels),
        "pretrained": (pretrained_acc, pretrained_labels),
        "test_size": test_size,
        "comparison": comparison_results,
    }


if __name__ == "__main__":
    args = get_args()

    #countries = ["France", "Germany", "Japan", "Russia", "United States"]
    countries = ["France", "Japan", "United States"]

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
    )

    # Generate random seed without a fixed seed
    random_seed = random.randint(0, 2**31 - 1)
    print(f"Using random seed: {random_seed}")

    try:
        # Collect results for each country
        all_results = {}
        for country in countries:
            if country != "United States":
                continue
            print(f"\n{'#'*60}")
            print(f"# Processing country: {country}")
            print(f"{'#'*60}")
            all_results[country] = asyncio.run(async_main(args, random_seed, country))

        # Aggregate results weighted by test set size
        total_test_size = sum(r["test_size"] for r in all_results.values())

        aggregated = {}
        for benchmark in ["icm", "golden", "chat", "pretrained"]:
            weighted_acc = sum(
                r[benchmark][0] * r["test_size"] for r in all_results.values()
            ) / total_test_size
            aggregated[benchmark] = weighted_acc

        # Print aggregated summary
        print("\n" + "="*60)
        print("AGGREGATED RESULTS (weighted by test set size)")
        print("="*60)
        print(f"Total test examples: {total_test_size}")
        print(f"ICM (Unsupervised):      {aggregated['icm']*100:.2f}%")
        print(f"Golden Supervision:      {aggregated['golden']*100:.2f}%")
        print(f"Zero-shot (Chat):        {aggregated['chat']*100:.2f}%")
        print(f"Zero-shot (Pretrained):  {aggregated['pretrained']*100:.2f}%")

        # Plot aggregated results
        plot_test_accuracies(
            aggregated["icm"],
            aggregated["golden"],
            aggregated["chat"],
            aggregated["pretrained"],
            "aggregated"
        )

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
                        'icm_acc_weighted': 0.0,
                        'random_acc_weighted': 0.0,
                        'icm_train_acc_weighted': 0.0,
                        'total_test_size': 0,
                    }
                comparison_by_num_examples[num_ex]['gold_acc_weighted'] += comparison['gold_acc'][i] * test_size
                comparison_by_num_examples[num_ex]['icm_acc_weighted'] += comparison['icm_acc'][i] * test_size
                comparison_by_num_examples[num_ex]['random_acc_weighted'] += comparison['random_acc'][i] * test_size
                comparison_by_num_examples[num_ex]['icm_train_acc_weighted'] += comparison['icm_train_acc'][i] * test_size
                comparison_by_num_examples[num_ex]['total_test_size'] += test_size

        # Build aggregated comparison results with weighted averages
        aggregated_comparison = {
            'num_examples': sorted(comparison_by_num_examples.keys()),
            'gold_acc': [],
            'icm_acc': [],
            'random_acc': [],
            'icm_train_acc': [],
        }
        for num_ex in aggregated_comparison['num_examples']:
            data = comparison_by_num_examples[num_ex]
            total = data['total_test_size']
            aggregated_comparison['gold_acc'].append(data['gold_acc_weighted'] / total)
            aggregated_comparison['icm_acc'].append(data['icm_acc_weighted'] / total)
            aggregated_comparison['random_acc'].append(data['random_acc_weighted'] / total)
            aggregated_comparison['icm_train_acc'].append(data['icm_train_acc_weighted'] / total)

        # Plot aggregated comparison
        plot_accuracy_vs_num_examples(aggregated_comparison, "Aggregated")

    finally:
        # Gracefully shutdown vLLM engine
        model_api.shutdown()
