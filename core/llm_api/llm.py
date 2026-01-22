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
    """
    Unified API for multiple LLM backends including OpenAI and vLLM.

    Supports:
    - OpenAI chat models via API
    - Self-hosted models via vLLM in-process inference
    """
    openai_fraction_rate_limit: float = attrs.field(
        default=0.99, validator=attrs.validators.lt(1)
    )
    print_prompt_and_response: bool = False

    # vLLM configuration
    use_vllm: bool = False
    vllm_model_name: Optional[str] = None
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.90
    vllm_max_model_len: Optional[int] = None
    vllm_max_num_batched_tokens: Optional[int] = None
    vllm_max_num_seqs: int = 485
    vllm_enable_prefix_caching: bool = True

    _openai_chat: OpenAIChatModel = attrs.field(init=False)
    _vllm_client: Optional[object] = attrs.field(init=False, default=None)

    # Cost and timing tracking
    running_cost: float = attrs.field(init=False, default=0.0)
    model_timings: dict = attrs.field(init=False, factory=dict)
    model_wait_times: dict = attrs.field(init=False, factory=dict)

    def __attrs_post_init__(self):
        secrets = load_secrets()

        # Initialize OpenAI client
        self._openai_chat = OpenAIChatModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            print_prompt_and_response=self.print_prompt_and_response,
        )
        assert self.vllm_model_name in ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B"]
        # Initialize vLLM client if enabled
        if self.use_vllm and self.vllm_model_name:
            from core.llm_api.vllm_llm import VLLMInProcessClient

            self._vllm_client = VLLMInProcessClient(
                model_name=self.vllm_model_name,
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                max_model_len=self.vllm_max_model_len,
                max_num_batched_tokens=self.vllm_max_num_batched_tokens,
                max_num_seqs=self.vllm_max_num_seqs,
                enable_prefix_caching=self.vllm_enable_prefix_caching,
                print_prompt_and_response=self.print_prompt_and_response,
            )

    def reset_cost(self):
        """Reset the running cost counter."""
        self.running_cost = 0.0

    def shutdown(self):
        """Gracefully shutdown vLLM engine and release GPU resources."""
        if self._vllm_client is not None:
            self._vllm_client.shutdown()
            self._vllm_client = None

    @staticmethod
    def _load_from_cache(save_file):
        if not os.path.exists(save_file):
            return None
        else:
            with open(save_file) as f:
                cache = json.load(f)
            return cache

    def _model_id_to_class(self, model_id: str) -> ModelAPIProtocol:
        """Route model requests to appropriate backend."""
        # Check if this is a vLLM model (base models, non-Instruct Llama, etc.)
        if self.use_vllm and self._vllm_client is not None:
            # Route base models to vLLM
            if "Instruct" not in model_id and "gpt" not in model_id.lower():
                return self._vllm_client

        # Route chat/instruct models to OpenAI API
        if model_id in GPT_CHAT_MODELS or "ft:gpt-3.5-turbo" in model_id:
            return self._openai_chat

        # For Instruct models, use OpenAI-compatible API (Together, etc.)
        if "Instruct" in model_id:
            return self._openai_chat

        # Default: if vLLM is enabled, use it for unknown models
        if self.use_vllm and self._vllm_client is not None:
            return self._vllm_client

        raise ValueError(f"Invalid model id: {model_id}")

    async def __call__(
        self,
        model_ids: Union[str, list[str]],
        prompt: Union[list[dict[str, str]], str, list[str]],
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 50,
        num_candidates_per_completion: int = 1,
        parse_fn=None,
        use_cache: bool = True,
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Make API requests for the specified model(s) and prompt.

        Supports:
        - Single prompt string
        - List of prompts (for batched vLLM inference)
        - Message list format (for chat models)

        Args:
            model_ids: The model(s) to call
            prompt: The prompt(s) to send
            print_prompt_and_response: Whether to print prompts/responses
            n: Number of completions per prompt
            max_attempts_per_api_call: Retry attempts
            num_candidates_per_completion: Candidates per completion
            parse_fn: Post-processing function for responses
            use_cache: Whether to use caching
            **kwargs: Additional arguments (max_tokens, temperature, logprobs, etc.)
        """
        assert (
            "max_tokens_to_sample" not in kwargs
        ), "max_tokens_to_sample should be passed in as max_tokens."

        if isinstance(model_ids, str):
            model_ids = [model_ids]

        model_class = self._model_id_to_class(model_ids[0])

        # Set default max_tokens if not specified
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 1

        # Call the appropriate backend
        responses = await model_class(
            model_ids,
            prompt,
            print_prompt_and_response,
            max_attempts_per_api_call,
            n=n,
            **kwargs,
        )

        # Apply parse function if provided
        modified_responses = []
        for response in responses:
            if parse_fn is not None:
                response = parse_fn(response)
            modified_responses.append(response)

        return modified_responses

    async def call_single(
        self,
        model_id: str,
        prompt: Union[str, list[dict]],
        **kwargs,
    ) -> LLMResponse:
        """Convenience method for single prompt, single response."""
        responses = await self(model_id, prompt, n=1, **kwargs)
        return responses[0]
