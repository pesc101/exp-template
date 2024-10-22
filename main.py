import datetime
import os
from pathlib import Path
import logging

import hydra
import omegaconf
import pandas as pd
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from tqdm import tqdm
from vllm import LLM, SamplingParams

import wandb
from config import Config
from src.llm.inference_runner import ChatInferenceRunner, OpenAIChatInferenceRunner

cs = ConfigStore.instance()
cs.store(name="task_name", node=Config)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_system_prompts(cfg: Config) -> str:
    """Load system prompts and Python functions from files."""
    sys_prompts = open(cfg.sys_prompt_path).read()
    if cfg.tools_file_path == "":
        return sys_prompts


def get_inference_runner(cfg: Config, sampling_params: SamplingParams):
    """Return appropriate inference runner based on the model name."""
    if cfg.model.model_name.startswith("gpt"):
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAIChatInferenceRunner(sampling_params, cfg.model.model_name, api_key)
    else:
        llm = LLM(
            model=cfg.model.model_name,
            gpu_memory_utilization=cfg.model.gpu_memory_utilization,
        )
        return ChatInferenceRunner(llm, sampling_params)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    load_dotenv(dotenv_path=Path("../.env"))
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        settings=wandb.Settings(start_method="thread"),
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        name=f"{datetime.datetime.now().strftime('%H:%M_%d-%m')}_{cfg.model.model_name_short}",
    )
    print(cfg)

    sampling_params = SamplingParams(
        temperature=cfg.model.temperature,
        max_tokens=cfg.model.max_tokens,
        top_p=cfg.model.top_p,
    )
    runner = get_inference_runner(cfg, sampling_params)
    sys_prompts = load_system_prompts(cfg)

    run.finish()


if __name__ == "__main__":
    main()
