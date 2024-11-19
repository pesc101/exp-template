"""Main script to run the evaluation pipeline."""

import datetime
import logging

import hydra
import omegaconf
from encourage.llm import BatchInferenceRunner
from encourage.metrics import (
    BLEU,
    F1,
    GLEU,
    ROUGE,
    AnswerRelevance,
    AnswerSimilarity,
    ContextLength,
    ContextRecall,
    ExactMatch,
    GeneratedAnswerLength,
    NonAnswerCritic,
    ReferenceAnswerLength,
)
from encourage.metrics.metric import Metric
from encourage.prompts import PromptCollection
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


@hydra.main(version_base=None, config_path="conf", config_name="defaults")
def main(cfg: Config) -> None:
    """Main function to run the evaluation pipeline."""
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
    sys_prompts = FileManager(cfg.sys_prompt_path).read()

    # Collect last "content" for "user" from each sublist
    last_user_contents = []
    for data in dataset.messages:
        last_user_content = None
        for item in data:
            if item["role"] == "user":
                last_user_content = item["content"]
        last_user_contents.append(last_user_content)

    context: list[dict[str, list]] = [
        {"contexts": [{"content": ctx[0]["text"]}]} for ctx in dataset.ctxs
    ]
    meta_data = [{"reference_answer": answer[0]} for answer in dataset.answers]

    prompt_collection = PromptCollection.create_prompts(
        sys_prompts,
        user_prompts=last_user_contents,
        contexts=context,
        meta_datas=meta_data,
        template_name=cfg.template_name,
    )

    ## Init Model and Run Inference
    sampling_params = SamplingParams(
        temperature=cfg.model.temperature,
        max_tokens=cfg.model.max_tokens,
        top_p=cfg.model.top_p,
    )

    llm = LLM(
        model=cfg.model.model_name,
        gpu_memory_utilization=cfg.model.gpu_memory_utilization,
    )
    runner = BatchInferenceRunner(llm, sampling_params)

    responses = runner.run(prompt_collection)
    json_dump = [response.to_dict() for response in responses.response_data]
    FileManager(cfg.output_file_path).dump_json(json_dump)

    metrics: list[Metric] = [
        AnswerRelevance(runner=runner),
        AnswerSimilarity(model_name="all-mpnet-base-v2"),
        NonAnswerCritic(runner=runner),
        ContextRecall(runner=runner),
        GeneratedAnswerLength(),
        ReferenceAnswerLength(),
        ContextLength(),
        BLEU(),
        GLEU(),
        ROUGE(rouge_type="rouge1"),
        ROUGE(rouge_type="rouge2"),
        ROUGE(rouge_type="rougeLsum"),
        ExactMatch(),
        F1(),
        # AnswerFaithfulness(runner=runner),
        # ContextPrecision(runner=runner),
        # MeanReciprocalRank(),
    ]

    results = {}
    for metric in metrics:
        print(f"Calculate: {metric.name}")
        results[metric.name] = metric(responses)

    print("=" * 30 + "\n" + "Evaluation report" + "\n" + "=" * 30)
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value.score}")
        wandb.log({metric_name: metric_value.score})

    if not cfg.debug:
        run.finish()


if __name__ == "__main__":
    main()
