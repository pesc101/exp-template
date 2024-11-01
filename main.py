import datetime
import logging
import os
from pathlib import Path

import hydra
import omegaconf
from dotenv import load_dotenv
from encourage.llm.inference_runner import (
    BatchInferenceRunner,
    ChatInferenceRunner,
    OpenAIChatInferenceRunner,
)
from encourage.prompts.prompt_collection import PromptCollection
from encourage.utils.file_manager import FileManager
from hydra.core.config_store import ConfigStore
from vllm import LLM, SamplingParams

import wandb
from config import Config
from src.data.hf import HuggingFaceDataset

cs = ConfigStore.instance()
cs.store(name="rag-eval", node=Config)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_inference_runner(cfg: Config, sampling_params: SamplingParams, batch: bool = False):
    """Return appropriate inference runner based on the model name."""
    if cfg.model.model_name.startswith("gpt"):
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAIChatInferenceRunner(sampling_params, cfg.model.model_name, api_key)

    llm = LLM(
        model=cfg.model.model_name,
        gpu_memory_utilization=cfg.model.gpu_memory_utilization,
    )

    if batch:
        return BatchInferenceRunner(llm, sampling_params)

    return ChatInferenceRunner(llm, sampling_params)


@hydra.main(version_base=None, config_path="conf", config_name="defaults")
def main(cfg: Config) -> None:
    load_dotenv(dotenv_path=Path("../.env"))
    if not cfg.debug:
        logger.setLevel(logging.DEBUG)
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            settings=wandb.Settings(start_method="thread"),
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            name=f"{datetime.datetime.now().strftime('%H:%M_%d-%m')}_{cfg.model.model_name_short}",
        )
        logger.info(cfg)

    dataset = HuggingFaceDataset(
        cfg.dataset.dataset_path,
        cfg.dataset.dataset_name,
        cfg.dataset.split,
        cfg.dataset.max_samples,
    )
    print(cfg.dataset.max_samples)
    print(len(dataset))
    sys_prompts = FileManager(cfg.sys_prompt_path).read()

    # Collect last "content" for "user" from each sublist
    last_user_contents = []
    for data in dataset.messages:
        last_user_content = None
        for item in data:
            if item["role"] == "user":
                last_user_content = item["content"]
        last_user_contents.append(last_user_content)

    context = [{"context": ctx[0]} for ctx in dataset.ctxs]
    meta_data = [{"answers": ctx} for ctx in dataset.answers]

    prompt_collection = PromptCollection.create_prompts(
        sys_prompts,
        user_prompts=last_user_contents,
        contexts=context,
        meta_datas=meta_data,
        template_name="prompt/llama3.1_rag.j2",
    )

    ## Init Model and Run Inference
    sampling_params = SamplingParams(
        temperature=cfg.model.temperature,
        max_tokens=cfg.model.max_tokens,
        top_p=cfg.model.top_p,
    )

    runner = get_inference_runner(cfg, sampling_params, batch=True)
    responses = runner.run(prompt_collection)
    json_dump = [response.to_dict() for response in responses.response_data]
    FileManager(cfg.output_file_path).dump_json(json_dump)

    if not cfg.debug:
        run.finish()


if __name__ == "__main__":
    main()
