import asyncio
import json
import random
from collections import Counter
from copy import deepcopy

import numpy as np


from src.model_querying.prompt_creation import (
    get_judge_prompt_fewshot,
)
from src.model_querying.solution_extraction import (
    extract_claim_logprobs,
)
from src.pipeline.pipeline import Pipeline, PipelineConfig
from src.tools.dataloaders import (
    load_assignments
)


def calculate_accuracy(train_data, inconsistent_pairs):
    return {
        "train_predict_distribution": Counter(
            [i["label"] for i in train_data.values()]
        ),
        "train_label_distribution": Counter(
            [i["vanilla_label"] for i in train_data.values()]
        ),
        "train_accuracy": np.mean(
            [i["label"] == i["vanilla_label"] for i in train_data.values()]
        ),
        "train_prob": np.mean(
            [
                i["score"] if i["label"] == 1 else -i["score"]
                for i in train_data.values()
            ]
        ),
        "train_size": len(train_data),
        "inconsistent_num": len(inconsistent_pairs),
    }


def update_assign_based_on_decision(data, decision):
    if decision["type"] == "contradiction":
        if decision["score"] > 0:
            data[decision["claim_1"]["uid"]]["label"] = 1
            data[decision["claim_2"]["uid"]]["label"] = 0
        else:
            data[decision["claim_1"]["uid"]]["label"] = 0
            data[decision["claim_2"]["uid"]]["label"] = 1
    return data


def pick_two_inconsistent_claims(data):
    claims = list(data.values())

    consistency_groups = {}
    for claim in claims:
        if claim['label'] is None:
            continue
        cid = claim["consistency_id"]
        if cid not in consistency_groups:
            consistency_groups[cid] = []
        consistency_groups[cid].append(claim)

    inconsistent_pairs = {}
    for group in consistency_groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                
                if  (group[i]["label"] == group[j]["label"]) and (group[i] == 1):
                    inconsistent_pairs[len(inconsistent_pairs)] = {
                        "claim_1": group[i],
                        "claim_2": group[j],
                        "consistency_id": group[i]["consistency_id"],
                        "type": "contradiction",
                    }
    random.shuffle(inconsistent_pairs)
    return inconsistent_pairs


def propose_consistencyfix(
    name=None,
    iter=None,
    assignment=None,
    use_cache=True,
):
    pipeline_name = f"propose-consistencyfix-iter-{iter}"
    if name is not None:
        pipeline_name += "-" + name
    pipeline_config = PipelineConfig(
        pipeline_name,
        openai_fraction_rate_limit=0.99,
        num_problems=None,
        use_cache=use_cache,
    )
    pipeline = Pipeline(pipeline_config)

    initial_assign = pipeline.add_load_data_step(
        "get_assign", load_assignments, assignment
    )

    pipeline.add_transformation_step(
        "pick_two_inconsistent_claims",
        pick_two_inconsistent_claims,
        dependencies=[initial_assign],
    )

    return pipeline

def run_consistencyfix(
    model,
    name=None,
    use_cache=True,
    decision_id=None,
    decision=None,
    iter=None,
    assignment=None,
    model_api=None,
):
    pipeline_name = f"consistencyfix-iter-{iter}"
    if decision_id is not None:
        pipeline_name += f"-{decision_id}"
    if name is not None:
        pipeline_name += "-" + name

    pipeline_config = PipelineConfig(
        pipeline_name,
        openai_fraction_rate_limit=0.99,
        num_problems=None,
        use_cache=use_cache,
    )
    pipeline = Pipeline(pipeline_config, model_api=model_api)

    assert assignment is not None
    initial_assign = pipeline.add_load_data_step(
        "get_assign", load_assignments, assignment
    )
    
    pick_claims = pipeline.add_transformation_step(
        "pick_two_inconsistent_claims",
        pick_two_inconsistent_claims,
        dependencies=[initial_assign],
    )

    def add_train_demonstrations(train_data):
        copy_data = deepcopy(train_data)
        keys = list(copy_data.keys())
        values = list(copy_data.values())
        saved_keys = [
            "prompt",
            "question",
            "choice",
            "consistency_id",
            "label",
            "vanilla_label",
        ]
        values = []
        for i in copy_data.values():
            values.append(
                {saved_key: i[saved_key] for saved_key in saved_keys if saved_key in i}
            )

        for idx, key in enumerate(keys):
            train_data[key]["demonstration"] = {
                prev_key: prev_value
                for j, (prev_key, prev_value) in enumerate(zip(keys, values))
                if j != idx
            }
        return train_data

    merged_train_data = pipeline.add_transformation_step(
        "add_train_demonstration",
        add_train_demonstrations,
        dependencies=[initial_assign],
    )

    get_train_preds = pipeline.add_batched_query_step(
        "get_train_preds",
        model,
        get_judge_prompt_fewshot,
        dependencies=[merged_train_data],
        logprobs=5,
        max_tokens=1,
        use_cache=use_cache,
    )

    eval_preds = pipeline.add_eval_step(
        "evaluate",
        calculate_accuracy,
        dependencies=[get_train_preds, pick_claims],
    )

    return pipeline