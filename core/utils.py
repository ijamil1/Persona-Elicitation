import asyncio
import json
import logging
import os
import time
from functools import lru_cache, wraps
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import replicate
import typer
import yaml
from tenacity import retry, retry_if_result, stop_after_attempt

typer.main.get_command_name = lambda name: name

LOGGER = logging.getLogger(__name__)
SEPARATOR = "---------------------------------------------\n\n"
SEPARATOR_CONVERSATIONAL_TURNS = "=============================================\n\n"
PROMPT_HISTORY = "prompt_history"
SECRETS_FILE_PATH = Path(__file__).parent.parent / "SECRETS"

LOGGING_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def setup_environment(
    logger_level: str = "info",
    openai_tag: str = "API_KEY",
    organization: str = None,
):
    setup_logging(logger_level)
    load_secrets(
        SECRETS_FILE_PATH,
        logger_level,
        openai_tag,
        organization,
    )


def setup_logging(level_str):
    level = LOGGING_LEVELS.get(
        level_str.lower(), logging.INFO
    )  # default to INFO if level_str is not found
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Disable logging from noisy libraries
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logging.getLogger("anthropic").setLevel(logging.CRITICAL)
    logging.getLogger("httpcore").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    LOGGER.info(f"Logging level set to {level_str}")


def load_secrets(
    file_path=SECRETS_FILE_PATH,
    logger_level: str = "info",
    openai_tag: str = "TOGETHER_API_KEY",
    organization: str = None,
):
    secrets = {}
    with open(file_path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            secrets[key] = value

    openai.api_key = secrets[openai_tag]

    if organization is not None:
        openai.organization = secrets[organization]
    if secrets.get("NEW_LLAMA_API_BASE") is not None:
        os.environ['NEW_LLAMA_API_BASE'] = secrets['NEW_LLAMA_API_BASE']
    return secrets


def load_yaml(file_path):
    with open(file_path) as f:
        content = yaml.safe_load(f)
    return content


@lru_cache(maxsize=8)
def load_yaml_cached(file_path):
    with open(file_path) as f:
        content = yaml.safe_load(f)
    return content


def save_yaml(file_path, data):
    with open(file_path, "w") as f:
        yaml.dump(data, f)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data


def save_jsonl(file_path, data):
    with open(file_path, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")


def typer_async(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # No event loop running
            loop = None

        if loop is None:
            return asyncio.run(f(*args, **kwargs))
        else:
            return f(*args, **kwargs)  # Return coroutine to be awaited

    return wrapper


@retry(
    stop=stop_after_attempt(16),
    retry=retry_if_result(lambda result: result is not True),
)
def function_with_retry(function, *args, **kwargs):
    return function(*args, **kwargs)


@retry(
    stop=stop_after_attempt(16),
    retry=retry_if_result(lambda result: result is not True),
)
async def async_function_with_retry(function, *args, **kwargs):
    return await function(*args, **kwargs)


