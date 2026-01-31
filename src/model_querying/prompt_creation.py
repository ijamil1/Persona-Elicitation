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
    """
    Create a few-shot prompt optimized for base/pretrained models.

    Uses minimal pattern completion format rather than instruction-following.
    """
    if demonstrations is None:
        demonstrations = list(example["demonstration"].values())

    country = example["country"]

    prompt = f"Survey Analysis: {country} Public Opinion\n\n"

    for demo in demonstrations:
        prompt += f"Survey question: \"{demo['question']}\"\n"
        prompt += f"Response option: \"{demo['choice']}\"\n"
        prompt += f"This response represents the majority opinion in {country}: {'True' if demo['label'] else 'False'}\n\n"

    prompt += f"Survey question: \"{example['question']}\"\n"
    prompt += f"Response option: \"{example['choice']}\"\n"
    prompt += f"This response represents the majority opinion in {country}: "

    if pipeline:
        return Prompt(prompt)
    else:
        return prompt


def get_judge_prompt_zeroshot(example, pipeline=True, is_chat_model=False):
    """
    Create a zero-shot prompt for evaluating a claim without demonstrations.

    Args:
        example: The example to evaluate
        pipeline: Whether to return a Prompt object or raw string
        is_chat_model: If True, use instruction-style prompt for chat/instruct models.
                       If False, use simple pattern completion for base models.
    """
    country = example["country"]

    if is_chat_model:
        # Instruction-style prompt for chat/instruct models
        prompt = f"You are an expert on the social, economic, and political dynamics of {country}. "
        prompt += f"Determine if the answer to a question (both listed below) represents the most common opinion among citizens of {country}. "
        prompt += "Respond with only 'True' or 'False'.\n\n"
        prompt += f"Q: {example['question']}\n"
        prompt += f"A: {example['choice']}\n"
        prompt += "Reflects majority view: "
    else:
        # Simple pattern completion for base/pretrained models (consistent with few-shot format)
        prompt = f"Survey Analysis: {country} Public Opinion\n\n"
        prompt += f"Survey question: \"{example['question']}\"\n"
        prompt += f"Response option: \"{example['choice']}\"\n"
        prompt += f"This response represents the majority opinion in {country}: "

    if pipeline:
        return Prompt(prompt)
    else:
        return prompt