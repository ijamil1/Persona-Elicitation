"""
vLLM In-Process Client for self-hosted LLM inference.

This module provides an in-process vLLM client that loads models directly into GPU memory,
eliminating network serialization overhead for maximum inference efficiency.
"""

import logging
from typing import Optional, Union

from core.llm_api.base_llm import LLMResponse, StopReason

LOGGER = logging.getLogger(__name__)

# Lazy imports to avoid loading vLLM unless needed
_vllm_module = None
_sampling_params_class = None


def _get_vllm():
    global _vllm_module
    if _vllm_module is None:
        import vllm
        _vllm_module = vllm
    return _vllm_module


def _get_sampling_params():
    global _sampling_params_class
    if _sampling_params_class is None:
        from vllm import SamplingParams
        _sampling_params_class = SamplingParams
    return _sampling_params_class


class VLLMInProcessClient:
    """
    In-process vLLM client for self-hosted inference.

    Loads the model directly into GPU memory and performs inference without
    network overhead. Supports batched inference for efficient processing.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        max_num_seqs: int = 485,
        enable_prefix_caching: bool = True,
        print_prompt_and_response: bool = False,
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.enable_prefix_caching = enable_prefix_caching
        self.print_prompt_and_response = print_prompt_and_response

        self._engine = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazily initialize the vLLM engine on first use."""
        if self._initialized:
            return

        LOGGER.info(f"Initializing vLLM engine with model: {self.model_name}")

        vllm = _get_vllm()

        engine_args = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_prefix_caching": self.enable_prefix_caching,
            "trust_remote_code": True,
        }

        if self.max_model_len is not None:
            engine_args["max_model_len"] = self.max_model_len

        if self.max_num_batched_tokens is not None:
            engine_args["max_num_batched_tokens"] = self.max_num_batched_tokens

        if self.max_num_seqs is not None:
            engine_args["max_num_seqs"] = self.max_num_seqs

        self._engine = vllm.LLM(**engine_args)
        self._initialized = True

        LOGGER.info("vLLM engine initialized successfully")

    def shutdown(self):
        """Gracefully release GPU resources."""
        if self._engine is not None:
            LOGGER.info("Shutting down vLLM engine")
            del self._engine
            self._engine = None
            self._initialized = False

            # Force GPU memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _generate_sync(
        self,
        prompts: list[str],
        max_tokens: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        logprobs: Optional[int] = None,
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Synchronous batch generation.

        Args:
            prompts: List of prompt strings
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            logprobs: Number of top logprobs to return per token

        Returns:
            List of LLMResponse objects
        """
        self._ensure_initialized()

        SamplingParams = _get_sampling_params()

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )

        if self.print_prompt_and_response:
            for i, prompt in enumerate(prompts):
                LOGGER.info(f"[vLLM Prompt {i}]: {prompt[:500]}...")

        # Batch generate
        outputs = self._engine.generate(prompts, sampling_params)

        responses = []
        for output in outputs:
            completion_text = output.outputs[0].text
            stop_reason = "stop" if output.outputs[0].finish_reason == "stop" else "length"

            # Extract logprobs if requested
            extracted_logprobs = None
            if logprobs is not None and output.outputs[0].logprobs:
                extracted_logprobs = []
                for token_logprobs in output.outputs[0].logprobs:
                    token_dict = {}
                    for token_id, logprob_obj in token_logprobs.items():
                        # vLLM returns Logprob objects with .decoded_token and .logprob
                        token_str = logprob_obj.decoded_token
                        token_dict[token_str] = logprob_obj.logprob
                    extracted_logprobs.append(token_dict)

            if self.print_prompt_and_response:
                LOGGER.info(f"[vLLM Response]: {completion_text}")

            response = LLMResponse(
                model_id=self.model_name,
                completion=completion_text,
                stop_reason=stop_reason,
                cost=0.0,  # Self-hosted = no per-token cost
                logprobs=extracted_logprobs,
            )
            responses.append(response)

        return responses

    async def __call__(
        self,
        model_ids: Union[str, list[str]],
        prompt: Union[str, list[str]],
        print_prompt_and_response: bool = False,
        max_attempts: int = 1,
        n: int = 1,
        max_tokens: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        logprobs: Optional[int] = None,
        top_logprobs: Optional[int] = None,
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Async entry point for vLLM inference.

        Accepts prompts in multiple formats:
        - Single string prompt
        - List of string prompts (batched inference)

        Args:
            model_ids: Model identifier(s) - ignored for vLLM (uses configured model)
            prompt: Single prompt string or list of prompts
            print_prompt_and_response: Whether to log prompts/responses
            max_attempts: Retry attempts (unused for local inference)
            n: Number of completions per prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            logprobs: Number of logprobs to return
            top_logprobs: Alias for logprobs parameter

        Returns:
            List of LLMResponse objects
        """
        # Handle logprobs parameter (support both names)
        actual_logprobs = logprobs or top_logprobs

        # Normalize prompt to list
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # vLLM generate is synchronous, but we wrap it for async interface
        responses = self._generate_sync(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=actual_logprobs,
        )

        return responses
