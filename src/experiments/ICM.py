import asyncio
import json
import math
import os
import random
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import argparse
from core.llm_api.llm import ModelAPI
from core.utils import setup_environment
from src.experiments.ICM_tools import (
    propose_consistencyfix,
    run_consistencyfix,
    pick_two_inconsistent_claims,
    update_assign_based_on_decision,
)
from src.model_querying.prompt_creation import (
    get_judge_prompt_fewshot
)
from src.model_querying.solution_extraction import (
    extract_claim_logprobs
)
from src.pipeline.pipeline import Pipeline, PipelineConfig
from src.tools.dataloaders import (
    load_assignments
)
from src.tools.path_utils import get_root_directory


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


def fix_inconsistency(demonstrations, cur_metric, name, iter=0, K=20):
    
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
        results = asyncio.run(pipeline.run())
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
                )
                tmp_results = asyncio.run(tmp_pipeline.run())
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

    for k in assignment:
        demonstrations[k] = assignment[k]

    return demonstrations, best_metric


def get_pipeline(
    model,
    name=None,
    use_cache=True,
    num_problems=None,
    decision_id=None,
    iter=None,
    assignment=None,
):
    pipeline_name = f"iterative-truth-assign-iter-{iter}"
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
    pipeline = Pipeline(pipeline_config)

    assert assignment is not None
    initial_assign = pipeline.add_load_data_step(
        "get_assign", load_assignments, assignment
    )

    def add_train_demonstrations(train_data):
        copy_data = deepcopy(train_data)
        copy_data = {k: v for k, v in copy_data.items() if v["label"] is not None}
        keys = list(copy_data.keys())
        values = list(copy_data.values())
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

    get_train_preds = pipeline.add_query_step(
        "get_train_preds",
        model,
        get_judge_prompt_fewshot,
        extract_claim_logprobs,
        dependencies=[merged_train_data],
        logprobs=20,
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
    demos = [
        v
        for k, v in demonstrations.items()
        if k != example["uid"] and v["label"] is not None
    ]
    model_requests = [
        model_api(
            model,
            get_judge_prompt_fewshot(
                example,
                demos,
                pipeline=False,
            ),
            logprobs=20,
            max_tokens=1,
            parse_fn=extract_claim_logprobs,
        )
    ]
    responses = await asyncio.gather(*model_requests)
    score = responses[0][0]["score"]
    new_label = score > 0
    return int(new_label)


def get_temperature(
    iteration, initial_temp, final_temp, decay_rate, schedule="exp"
):
    """
    Calculate the temperature for simulated annealing.

    Parameters:
    - iteration: Current iteration number.
    - initial_temp: Initial temperature.
    - decay_rate: Rate at which the temperature decreases.

    Returns:
    - Current temperature.
    """
    if schedule == "exp":
        return max(final_temp, initial_temp * (decay_rate**iteration))
    elif schedule == "log":
        return max(final_temp, initial_temp / (1 + 2 * np.log(1 + iteration)))
    else:
        assert False


def get_energy(metric, alpha):
    return alpha * metric["train_prob"] - metric["inconsistent_num"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=30)
    parser.add_argument("--seed", type=int, default=27565976)
    parser.add_argument("--testbed", type=str, default="gsm8k")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B")
    parser.add_argument("--num_seed", type=int, default=8)
    parser.add_argument("--K", type=int, default=3000)
    parser.add_argument("--consistency_fix_K", type=int, default=10)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--initial_T", type=float, default=10)
    parser.add_argument("--final_T", type=float, default=0.01)
    parser.add_argument("--scheduler", type=str, default="log")
    parser.add_argument("--country", type=str, default="United States")
    args = parser.parse_args()
    return args

def load_data(args):

    with open(get_root_directory() / "data/transformed_global_opinions.json") as f:
        data = json.load(f)

    # Filter by country
    country_data = [item for item in data if item['country'] == args.country]

    template = """Question: {question}
        Claim: {answer}
        I think this claim is """

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
    random.shuffle(group_ids)

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

def initialize(train, fewshot_ids, args):
    demonstrations = {}
    whole_ids = []

    random_init_labels = [1] * (args.num_seed // 2) + [0] * (args.num_seed // 2)
    random.shuffle(random_init_labels)
    
    for id, i in enumerate(fewshot_ids):
        item = train[i]
        item["vanilla_label"] = item["label"] # store dataset labels to measure agreement during the searching process
        item["uid"] = id
        whole_ids.append(item["uid"])
        if id >= args.num_seed:  # set labels to None
            item["label"] = None
            item["type"] = "predict"
        else: # set random labels
            item["type"] = "seed"
            item["label"] = random_init_labels[id]
        demonstrations[id] = item
        
    return demonstrations, whole_ids


def icm_main(args, train, fewshot_ids):
    demonstrations, whole_ids = initialize(train, fewshot_ids, args)

    cur_metric = {
        "train_prob": -1e6,
        "inconsistent_num": 100000,
        "train_accuracy": 1.0,
        "train_predict_distribution": {"0": 0, "1": 0},
        "train_label_distribution": {"0": 0, "1": 0},
    }

    print(f'Country: {args.country}, Train size: {len(train)}')
    print('init random labels = ', Counter([i['label'] for i in demonstrations.values() if i['type'] == 'seed']), 'init label acc = ', np.mean([i['label'] == i['vanilla_label'] for i in demonstrations.values() if i['type'] == 'seed']))
    name = f"{args.country}-K{args.K}_seed{args.seed}-initialsize{args.num_seed}-weighted{args.alpha}-decay{args.decay}-initialT{args.initial_T}-finalT{args.final_T}-scheduler{args.scheduler}"

    iter = 0
    flip_cnt = 0
    example_id = 0

    for _ in tqdm(range(args.K), desc="searching"):
        cur_pool = {
            k: v for k, v in demonstrations.items() if v["label"] is not None
        }
       
        if iter == 0:
            pipeline = get_pipeline(
                args.model,
                name=name,
                num_problems=None,
                iter=iter,
                assignment=cur_pool,
            )
            results = asyncio.run(pipeline.run())
            cur_metric = results["evaluate"]
            
            demonstrations, cur_metric = fix_inconsistency(
                demonstrations, cur_metric, name, iter=iter, K=args.consistency_fix_K
            )
            
        cur_pool = {
            k: v for k, v in demonstrations.items() if v["label"] is not None
        }
       
        while True: # weighted sampling
            candidates_ids = whole_ids
            weights = [1 for _ in range(len(candidates_ids))]
            for i in candidates_ids:
                if i in cur_pool:
                    same_consistency_group_ids = [j for j in candidates_ids if demonstrations[j]["consistency_id"] == demonstrations[i]["consistency_id"]]
                    for j in same_consistency_group_ids:
                        if j not in cur_pool:
                            weights[j] = 100

            example_id = random.choices(candidates_ids, k=1, weights=weights)[0]
            break

        new_label = asyncio.run(
            predict_assignment(
                args.model,
                demonstrations[example_id],
                cur_pool,
            )
        )

        if demonstrations[example_id]["label"] != new_label:
            tmp_demonstrations = deepcopy(demonstrations)
            tmp_demonstrations[example_id]["label"] = new_label
            dummy_metric = {
                "train_prob": -1e6,
                "inconsistent_num": 100000,
                "train_accuracy": 1.0,
                "train_predict_distribution": {"0": 0, "1": 0},
                "train_label_distribution": {"0": 0, "1": 0},
            }

            tmp_demonstrations, _ = fix_inconsistency(
                tmp_demonstrations,
                dummy_metric,
                name + "newlabelexplore",
                iter=iter,
                K=10,
            )
            
            tmp_pool = {
                k: v
                for k, v in tmp_demonstrations.items()
                if v["label"] is not None
            }
            pipeline = get_pipeline(
                model=args.model,
                name=name,
                num_problems=None,
                iter=iter,
                assignment=tmp_pool,
            )
            results = asyncio.run(pipeline.run())
            metric = results["evaluate"]
            T = get_temperature(
                flip_cnt, args.initial_T, args.final_T, args.decay, schedule=args.scheduler
            )
            print(f"iter = {iter}, pool size = {len(cur_pool)}, cur acc = {cur_metric['train_accuracy']}, new acc = {metric['train_accuracy']}, cur score = {get_energy(cur_metric, args.alpha)}, new score = {get_energy(metric, args.alpha)}, cur inconsistent num = {cur_metric['inconsistent_num']}, new inconsistent num = {metric['inconsistent_num']}")

            accept_prob = math.exp((get_energy(metric, args.alpha) - get_energy(cur_metric, args.alpha)) / T)
            if random.random() < accept_prob:
                demonstrations = tmp_demonstrations
                flip_cnt += 1
                cur_metric = metric
                
                #with open(f"log_{name}.jsonl", "a") as f:
                #    f.write(json.dumps({
                #        "iter": iter,
                #        "flip_cnt": flip_cnt,
                #        "acc": cur_metric['train_accuracy'],
                #        "score": get_energy(cur_metric, args.alpha),
                #    }) + "\n")

        print("=" * 100)
        iter += 1


if __name__ == "__main__":
    setup_environment(logger_level="error")
    model_api = ModelAPI(openai_fraction_rate_limit=0.99)
    args = get_args()
    random.seed(args.seed)
    train, fewshot_ids, test = load_data(args)
    icm_main(args, train, fewshot_ids)