import asyncio
import json
import logging
import os
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import attrs

from core.llm_api.base_llm import LLMResponse, ModelAPIProtocol
from core.llm_api.openai_llm import (
    GPT_CHAT_MODELS,
    OpenAIChatModel,
)
from core.utils import load_secrets

LOGGER = logging.getLogger(__name__)


@attrs.define()
class ModelAPI:
    openai_fraction_rate_limit: float = attrs.field(
        default=0.99, validator=attrs.validators.lt(1)
    )
    print_prompt_and_response: bool = False
    _openai_chat: OpenAIChatModel = attrs.field(init=False)

    def __attrs_post_init__(self):
        secrets = load_secrets()
        self._openai_chat = OpenAIChatModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=secrets[self.organization],
            print_prompt_and_response=self.print_prompt_and_response,
        )

    @staticmethod
    def _load_from_cache(save_file):
        if not os.path.exists(save_file):
            return None
        else:
            with open(save_file) as f:
                cache = json.load(f)
            return cache

    async def __call__(
        self,
        model_ids: Union[str, list[str]],
        prompt: Union[list[dict[str, str]], str],
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 50,
        num_candidates_per_completion: int = 1,
        parse_fn=None,
        use_cache: bool = True,
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Make maximally efficient API requests for the specified model(s) and prompt.

        Args:
            model_ids: The model(s) to call. If multiple models are specified, the output will be sampled from the
                cheapest model that has capacity. All models must be from the same class (e.g. OpenAI Base,
                OpenAI Chat, or Anthropic Chat). Anthropic chat will error if multiple models are passed in.
                Passing in multiple models could speed up the response time if one of the models is overloaded.
            prompt: The prompt to send to the model(s). Type should match what's expected by the model(s).
            max_tokens: The maximum number of tokens to request from the API (argument added to
                standardize the Anthropic and OpenAI APIs, which have different names for this).
            print_prompt_and_response: Whether to print the prompt and response to stdout.
            n: The number of completions to request.
            max_attempts_per_api_call: Passed to the underlying API call. If the API call fails (e.g. because the
                API is overloaded), it will be retried this many times. If still fails, an exception will be raised.
            num_candidates_per_completion: How many candidate completions to generate for each desired completion. n*num_candidates_per_completion completions will be generated, then is_valid is applied as a filter, then the remaining completions are returned up to a maximum of n.
            parse_fn: post-processing on the generated response
            save_path: cache path
            use_cache: whether to load from the cache or overwrite it
        """

        assert (
            "max_tokens_to_sample" not in kwargs
        ), "max_tokens_to_sample should be passed in as max_tokens."

        if isinstance(model_ids, str):
            model_ids = [model_ids]
            # # trick to double rate limit for most recent model only

        def model_id_to_class(model_id: str) -> ModelAPIProtocol:
            if model_id in ["gpt-4-base", "gpt-3.5-turbo-instruct"]:
                return (
                    self._openai_base_arg
                )
            elif model_id in GPT_CHAT_MODELS or "ft:gpt-3.5-turbo" in model_id:
                return self._openai_chat
            raise ValueError(f"Invalid model id: {model_id}")

        model_classes = [model_id_to_class(model_id) for model_id in model_ids]
       
        if len(set(str(type(x)) for x in model_classes)) != 1:
            raise ValueError("All model ids must be of the same type.")

        model_class = model_classes[0]
       
        kwargs["max_tokens"] = 1
            
        responses = await model_class(
            model_ids,
            prompt,
            print_prompt_and_response,
            max_attempts_per_api_call,
            n=1,
            **kwargs,
        )
        modified_responses = []
        for response in responses:
            if parse_fn is not None:
                response = parse_fn(response)
            modified_responses.append(response)
        return modified_responses


