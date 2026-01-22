__all__ = ["PipelineConfig", "Pipeline"]

import asyncio
import logging
from collections import deque
from typing import Optional

from tqdm.auto import tqdm

from core.llm_api.llm import ModelAPI
from src.datatypes.enums import Language
from src.tools.dataloaders import read_from_cache, save_to_cache
from src.model_querying.solution_extraction import get_yes_no_diff_logprobs

logger = logging.getLogger(__name__)

class Task:
    def __init__(self, name, func, use_cache, dependencies=[]):
        self.name = name
        self.func = func
        self.use_cache = use_cache
        self.index = None
        self.dependencies = dependencies
        self.dependents = []
        self.result = None
        for dep in dependencies:
            dep.dependents.append(self)

    async def execute(self, results):
        if self.result is None:
            dep_results = [results[dep.name] for dep in self.dependencies]
            if asyncio.iscoroutinefunction(self.func):
                self.result = await self.func(
                    *dep_results, use_cache=self.use_cache, index=self.index
                )
            else:
                self.result = self.func(
                    *dep_results, use_cache=self.use_cache, index=self.index
                )
        return self.result


class PipelineConfig:
    def __init__(
        self,
        name,
        openai_fraction_rate_limit=0.99,
        use_cache=True,
        language=Language.PYTHON,
        num_problems=None,
        problem_ids=None,
        num_open_files=1000000,
        print_prompt_and_response=False,
        api_base=None,
    ):
        self.name = name
        self.openai_fraction_rate_limit = openai_fraction_rate_limit
        self.print_prompt_and_response = print_prompt_and_response
        self.use_cache = use_cache
        self.language = language
        self.num_problems = num_problems
        self.problem_ids = problem_ids
        self.num_open_files = num_open_files
        self.api_base = api_base


class Pipeline:
    def __init__(self, config, model_api: Optional[ModelAPI] = None):
        """
        Initialize Pipeline.

        Args:
            config: PipelineConfig instance
            model_api: Optional shared ModelAPI instance. If not provided, creates a new one.
                       Pass a shared instance to use vLLM or to share API clients across pipelines.
        """
        self.config = config
        self.steps = []
        self.step_names = set()
        self.results = {}

        # Use provided model_api or create a new one
        if model_api is not None:
            self.model_api = model_api
        else:
            self.model_api = None

        self.file_sem = asyncio.Semaphore(self.config.num_open_files)

    def add_load_data_step(
        self, name, dataloader_fn, data_location, dependencies=[], use_cache=None
    ):
        if name in self.step_names:
            raise ValueError(f"Step name {name} already exists")
        self.step_names.add(name)

        def call(*args, use_cache, index):
            return dataloader_fn(
                data_location,
                num_problems=self.config.num_problems,
                problem_ids=self.config.problem_ids,
            )

        task = Task(name, call, use_cache, dependencies)
        self.steps.append(task)
        return task

    def add_batched_query_step(
        self,
        name,
        model,
        prompt_fn,
        dependencies=[],
        use_cache=None,
        temperature=0.0,
        logprobs=5,
        max_tokens=1,
    ):
        """
        Add a batched query step optimized for vLLM inference.

        Instead of making N separate API calls, this creates a single batched call
        that processes all examples together using vLLM's batch inference.

        Args:
            name: Step name
            model: Model identifier
            prompt_fn: Function to generate prompt from example
            dependencies: List of dependent steps
            use_cache: Whether to use caching
            temperature: Sampling temperature
            logprobs: Number of logprobs to return
            max_tokens: Maximum tokens to generate
        """
        if name in self.step_names:
            raise ValueError(f"Step name {name} already exists")
        self.step_names.add(name)

        async def call(data, use_cache, index):
            """
            Perform batched inference on all examples.

            Args:
                data: Dict of {example_id: example_data}
                      Each example must have fields needed by prompt_fn

            Returns:
                Dict of {example_id: example_with_score}
            """
            example_ids = list(data.keys())
            examples = list(data.values())

            # Prepare all prompts at once
            all_prompts = []
            for example in examples:
                prompt = prompt_fn(example)
                # Handle Prompt objects
                if hasattr(prompt, 'text'):
                    prompt = prompt.text
                all_prompts.append(prompt)

            # Make a single batched request
            # vLLM handles batching internally when given a list of prompts
            responses = await self.model_api(
                model,
                all_prompts,
                logprobs=logprobs,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Extract scores from batched responses
            result = {}
            for example_id, example, response in zip(example_ids, examples, responses):
                example_copy = example.copy()
                try:
                    # Handle LLMResponse objects
                    if hasattr(response, 'logprobs'):
                        logprobs_data = response.logprobs
                    else:
                        logprobs_data = response.get("logprobs", response.get("response", {}).get("logprobs"))

                    score = get_yes_no_diff_logprobs(logprobs_data)
                    example_copy["score"] = score
                except Exception as e:
                    logger.warning(f"Error extracting score for example {example_id}: {e}")
                    example_copy["score"] = 0

                result[example_id] = example_copy

            return result

        step = Task(name, call, use_cache, dependencies)
        self.steps.append(step)
        return step

    def add_transformation_step(
        self,
        name,
        transformation_fn,
        dependencies=[],
        use_cache=None,
        strong_model=None,
        weak_model=None,
        read_cache=False,
    ):
        if name in self.step_names:
            raise ValueError(f"Step name {name} already exists")
        self.step_names.add(name)

        async def call(*args, use_cache, index):
            incoming_problem_ids = set().union(*[arg.keys() for arg in args])
            if use_cache and read_cache:
                logger.debug(
                    f"Reading from cache for transformation: {self.config.name}/{index:02d}-{name}/{strong_model}{'+' if strong_model and weak_model else ''}{weak_model}"
                )
                data, cached_problem_ids = read_from_cache(
                    f"{self.config.name}/{index:02d}-{name}/{strong_model}{'+' if strong_model and weak_model else ''}{weak_model}"
                )
                if incoming_problem_ids.issubset(set(cached_problem_ids)):
                    return {k: v for k, v in data.items() if k in incoming_problem_ids}

            if asyncio.iscoroutinefunction(transformation_fn):
                output = await transformation_fn(*args)
            else:
                output = transformation_fn(*args)

            async with self.file_sem:
                save_to_cache(
                    output,
                    f"{self.config.name}/{index:02d}-{name}/{strong_model}{'+' if strong_model and weak_model else ''}{weak_model}",
                    delete_existing=read_cache,
                    incoming_problem_ids=incoming_problem_ids,
                )
            return output

        step = Task(name, call, use_cache, dependencies)
        self.steps.append(step)
        return step

    def add_eval_step(
        self,
        name,
        eval_fn,
        dependencies=[],
        strong_model=None,
        weak_model=None,
    ):
        if name in self.step_names:
            raise ValueError(f"Step name {name} already exists")
        self.step_names.add(name)

        async def call(*args, use_cache, index):
            output = eval_fn(*args)
            cache_obj = {"summary": output}
            async with self.file_sem:
                save_to_cache(
                    cache_obj,
                    f"{self.config.name}/{index:02d}-{name}/{strong_model}{'+' if strong_model and weak_model else ''}{weak_model}",
                )
            return output

        step = Task(name, call, None, dependencies)
        self.steps.append(step)
        return step

    def add_cost_data(self, team, response_dict):
        """Track costs from API calls."""
        pass  # Placeholder for cost tracking

    def speak(self, message):
        """Log a message."""
        logger.info(message)

    def topological_sort_tasks(self, tasks):
        in_degree = {task: len(task.dependencies) for task in tasks}

        queue = deque([task for task in tasks if in_degree[task] == 0])
        sorted_tasks = []
        task_order = {task: i for i, task in enumerate(tasks)}

        while queue:
            task = queue.popleft()
            sorted_tasks.append(task)
            for dependent in task.dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
                    queue = deque(sorted(queue, key=lambda t: task_order[t]))

        for i, task in enumerate(sorted_tasks):
            task.index = i
        return sorted_tasks

    def set_use_cache(self, tasks):
        # This is called after tasks.sort, so we are guaranteed to process all
        # dependencies before each task itself.
        for task in tasks:
            if not self.config.use_cache:
                task.use_cache = False
                continue

            if task.use_cache is None:
                task.use_cache = True

            for dep in task.dependencies:
                if not dep.use_cache:
                    task.use_cache = False

    async def run(self):
        steps = self.topological_sort_tasks(self.steps)
        self.set_use_cache(steps)
        for task in steps:
            logger.info(
                f"Starting step {task.index}: {task.name} - Using cache: {task.use_cache}"
            )
            try:
                self.results[task.name] = await task.execute(self.results)
            except Exception as e:
                logger.error(f"Error in step {task.index}: {task.name}")
                logger.error(e)
                self.speak("Pipeline failed sad face")
                raise e
            logger.info(f"Finished step {task.index}: {task.name}")
        self.speak("Jobs done")
        logger.info("Run complete!! Nice!!")
        return self.results
