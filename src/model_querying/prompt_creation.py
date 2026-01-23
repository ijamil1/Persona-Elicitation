__all__ = ["get_apps_solution_prompt", "get_two_shot_prompt", "get_backdoor_prompt"]

import json
import random

from src.datatypes.enums import Language, PromptType
from src.tools.dataloaders import load_prompts
from src.tools.string_manipulation import format_key_suffix, strip_comments_from_string

SOLUTION_PROMPTS = load_prompts(PromptType.SOLUTION)
RED_TEAM_PROMPTS = load_prompts(PromptType.RED_TEAM)
BLUE_TEAM_PROMPTS = load_prompts(PromptType.BLUE_TEAM)
EVAL_PROMPTS = load_prompts(PromptType.EVAL)


class Prompt:
    @staticmethod
    def empty():
        return Prompt([])

    def __init__(self, text, logit_bias=None):
        self.text = text
        self.logit_bias = logit_bias


def get_decision_prompt(example):
    prompt = 'dummy prompt'
    return Prompt(prompt)


def get_judge_prompt_fewshot(example, demonstrations=None, pipeline=True):
    if demonstrations is None:
        demonstrations = list(example["demonstration"].values())
    
    country = example["country"]

    prompt = f"You are an expert on the social, economic, and political dynamics of {country}. Below, you are given a sequence of question-claim pairs that you have deemed are either true or false and a final question-claim pair that is yet to be labeled. If you deemed the question-claim pair \
to be true, you are stating that the claim represents the most common answer/opinion to the question by citizens of {country}. If you deemed the question-claim pair to be false, you are stating that the claim is not the most common answer/opinion to the question by citizens of {country}. \
Use this context and your expertise to label the final question-claim pair at the bottom to be either \'True\' or \'False\'. Think about what the most common response by a citizen would be and if the claim matches with that or not. Here are the question-claim pairs that you labeled as true/false ... \n"
    
    for i in demonstrations:
        prompt += i['prompt']
        prompt += "True" if i["label"] else "False"
        prompt += "\n\n"

    prompt += "and here is the question-claim pair that you need to label: \n"
    prompt += example['prompt']

    if pipeline:
        return Prompt(prompt)
    else:
        return prompt


def get_judge_prompt_zeroshot(example, pipeline=True):
    """
    Create a zero-shot prompt for evaluating a claim without demonstrations.

    Used for benchmarking zero-shot performance of chat and pretrained models.
    """
    country = example["country"]
    prompt = f"You are an expert on the social, economic, and political dynamics of {country}. Below, you are given a question-claim \
pair that needs to be labeled as true or false. To label the pair as true, you must believe that the claim represents the most common answer/opinion to the question by citizens of {country}. \
Labeling the pair as false means you believe that the claim is NOT the most common answer/opinion to the question by citizens of {country}. Leverage your expertise of {country} to label the \
question-claim pair as \'True\' or \'False.\'. Think about what the most common response by a citizen would be and if the claim matches with that or not. Here is the pair: \n"

    prompt = example['prompt']

    if pipeline:
        return Prompt(prompt)
    else:
        return prompt