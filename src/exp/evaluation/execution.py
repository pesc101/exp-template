"""Module for evaluation of QA datasets."""

from pathlib import Path

import hydra
import litellm
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from encourage.llm import BatchInferenceRunner
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context
from encourage.prompts.meta_data import MetaData
from vllm import SamplingParams

from exp.evaluation.config import Config
from exp.evaluation.evaluation import main as evaluation
from exp.utils.file_manager import FileManager
from exp.utils.flatten_dict import flatten_dict

config_path = str((Path(__file__).parents[3] / "conf").resolve())


def prepare_data(df: pd.DataFrame) -> tuple[list, list]:
    """Prepare data for the inference runner."""
    meta_datas = []
    contexts = []
    for _, row in df.iterrows():
        meta_data = MetaData(
            {
                "figure_id": row["figure_id"],
                "image_path": row["image_file"],
                "question": row["question"],
                "reference_answer": row["answer"],
                "caption": row["caption"],
                "qa_pair_type": row["qa_pair_type"],
                "categories": row["categories"],
            }
        )
        meta_datas.append(meta_data)
        context = Context.from_prompt_vars(
            {
                "qa_pair_type": row["qa_pair_type"],
                "answer_options": {
                    k: v for d in row["answer_options"] for k, v in d.items() if v is not None
                },
            }
        )
        contexts.append(context)
    return meta_datas, contexts


@hydra.main(version_base=None, config_path=config_path, config_name="defaults")
def main(cfg: Config) -> None:
    """Main function for evaluation of QA datasets."""
    # Load dataset from Huggingface
    load_dotenv(".env")
    qa_dataset = load_dataset(cfg.dataset.name, split=cfg.dataset.split).to_pandas()

    litellm._logging._disable_debugging()
    mlflow.openai.autolog()

    sampling_params = SamplingParams(
        temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens
    )
    runner = BatchInferenceRunner(sampling_params, cfg.model.model_name, base_url=cfg.base_url)
    sys_prompt = FileManager(cfg.dataset.sys_prompt_path).read()

    ## Run the Inference
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(experiment_name=cfg.mlflow.experiment_id)

    with mlflow.start_run():
        mlflow.log_params(flatten_dict(cfg))
        mlflow.log_params({"dataset_size": len(qa_dataset)})
        mlflow.log_input(
            mlflow.data.pandas_dataset.from_pandas(
                qa_dataset.drop(columns=["answer_options"]), name=cfg.dataset.name
            ),
            context="inference",
        )

        with mlflow.start_span(name="root"):
            image_paths = [
                [f"{cfg.dataset.input_dir}/{cfg.dataset.split}/{qa_dataset['image_file'][i]}"]
                for i in range(len(qa_dataset))
            ]

            meta_datas, contexts = prepare_data(qa_dataset)

            prompt_collection = PromptCollection.create_image_prompts(
                sys_prompts=sys_prompt,
                user_prompts=qa_dataset["question"].tolist(),
                image_paths=image_paths,
                meta_datas=meta_datas,
                contexts=contexts,
                template_name=cfg.template_name,
            )
            responses = runner.run(prompt_collection)

        # Save the output to hydra folder0
        json_dump = [response.to_dict() for response in responses.response_data]
        FileManager(cfg.output_folder + "/inference_log.json").dump_json(
            json_dump, pydantic_encoder=True
        )
        json_dump = [flatten_dict(response.to_dict()) for response in responses.response_data]

        active_run = mlflow.active_run()
        run_name = active_run.info.run_name if active_run else "responses"
        mlflow.log_table(data=pd.DataFrame(json_dump), artifact_file=f"{run_name}.json")

        # Evaluate the retrieval
        evaluation(cfg)


if __name__ == "__main__":
    main()
