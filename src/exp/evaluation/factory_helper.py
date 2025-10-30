"""Helper functions for the evaluation factory."""

from typing import Any, Dict, Optional, Tuple

from encourage.llm import BatchInferenceRunner
from encourage.metrics import METRIC_REGISTRY, get_metric_from_registry
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from exp.evaluation.config import Config


def load_metrics(
    config: list[str | dict[str, dict[str, str]]], runner: BatchInferenceRunner = None
) -> list:
    """Load metrics from the config."""
    metrics = []
    for m in config:
        if isinstance(m, str):
            name, args = m, {}  # type: ignore
        elif isinstance(m, (dict, DictConfig)):
            # Convert DictConfig to plain dict
            m = OmegaConf.to_container(m, resolve=True)  # type: ignore
            name, args = next(iter(m.items()))
        else:
            raise ValueError(f"Invalid metric config: {m}")

        cls = METRIC_REGISTRY[name.lower()]

        if cls.requires_runner():
            metric = get_metric_from_registry(name, runner=runner, **args)
        else:
            metric = get_metric_from_registry(name, **args)

        metrics.append(metric)
    return metrics


def get_response_format(cfg: Config) -> Optional[type[BaseModel]]:
    """Get the response model from the config."""
    if not hasattr(cfg, "dataset") or not hasattr(cfg.dataset, "response_format"):
        return None
    fields: Dict[str, Tuple[Any, Any]] = {
        k: (eval(v) if isinstance(v, str) else v, ...)
        for k, v in cfg.dataset.response_format.items()
    }
    return create_model("ResponseModel", **fields)
