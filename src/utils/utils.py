"""Utilities for data processing."""

import os
from dataclasses import asdict, is_dataclass
from typing import Any

from dotenv import load_dotenv

from config import Config

load_dotenv()  # Load .env


def get_secret(key: str) -> str:
    """Get secrets saved in env files."""
    current_dir = os.path.dirname(__file__)
    env_secret_path = os.path.join(current_dir, ".env")

    load_dotenv(dotenv_path=env_secret_path)  # Load .env.secret

    # Retrieve the value of the key
    value = os.getenv(key)

    if value is None:
        raise KeyError(f"Secret '{key}' not found in the environment variables.")

    return value


def _process_values(value: Any) -> dict[str, str | int | float]:
    if value is None:
        return {"": "None"}
    if isinstance(value, bool):
        return {"": str(value)}
    if isinstance(value, (str, int, float)):
        return {"": value}
    if isinstance(value, list):
        return {
            f"{i}{k0}": v0 for i, v in enumerate(value) for k0, v0 in _process_values(v).items()
        }
    try:
        return {
            f"_{key}{k0}": v0 for key, v in value.items() for k0, v0 in _process_values(v).items()
        }
    except AttributeError as e:
        raise ValueError(f"Can not convert value {value} to scalar") from e


def flatten_dict(config: dict | Config | Any) -> dict[str, str | int | float]:
    """Convert a nested config to a flat dictionary suitable for TensorBoard logging.

    :param config: A dict or Config instance
    :return: A flat dictionary with keys of the form
        "outer_middle_inner".
    :raises ValueError: If values of config are not of type str, int,
        float or bool.
    """
    # Convert Config to a dictionary if it is a dataclass
    if is_dataclass(config):
        config = asdict(config)  # type: ignore
    elif not isinstance(config, dict):
        try:
            config = dict(config)
        except TypeError:
            raise ValueError("The provided config must be a dictionary or a dataclass.") from None

    # Flatten the dictionary
    return {
        f"{key}{k}": v for key, value in config.items() for k, v in _process_values(value).items()
    }
