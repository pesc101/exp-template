import datetime
import logging
import os
from pathlib import Path

import hydra
import omegaconf
import wandb
from dotenv import load_dotenv
from encourage.handler import ConversationHandler
from encourage.llm import ChatInferenceRunner, OpenAIChatInferenceRunner
from encourage.utils import FileManager
from hydra.core.config_store import ConfigStore
from vllm import LLM, SamplingParams

from src.exp.evaluation.config import Config

cs = ConfigStore.instance()
cs.store(name="task_name", node=Config)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


@hydra.main(version_base=None, config_path="conf", config_name="defaults")
def main(cfg: Config) -> None:
    """Main function to run the conversation handler."""
    load_dotenv(dotenv_path=Path("../.env"))
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        settings=wandb.Settings(start_method="thread"),
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        name=f"{datetime.datetime.now().strftime('%H:%M_%d-%m')}_{cfg.model.model_name_short}",
    )
    logger.info(cfg)

    sys_prompts = open(cfg.sys_prompt_path).read()
    sampling_params = SamplingParams(
        temperature=cfg.model.temperature,
        max_tokens=cfg.model.max_tokens,
        top_p=cfg.model.top_p,
    )
    runner = get_inference_runner(cfg, sampling_params)

    # Create an Conversation with ConversationHandler
    handler = ConversationHandler(
        runner, sys_prompts, ["How are you?"], template_name="prompt/test_template.j2"
    )
    responses = handler.run()
    FileManager(cfg.output_file_path + "inference_log.json").dump_json(responses.to_output())

    run.finish()


if __name__ == "__main__":
    main()
