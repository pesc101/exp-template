"""Module for evaluation of QA datasets."""

from pathlib import Path

import hydra
import hydra.core.hydra_config
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.rag import RAGFactory
from vllm import SamplingParams

from data.finqa_qa import FinQADatasetCollection
from exp.evaluation.config import Config
from exp.evaluation.evaluation import main as evaluation
from exp.utils.file_manager import FileManager
from exp.utils.flatten_dict import flatten_dict

config_path = str((Path(__file__).parents[3] / "conf").resolve())


@hydra.main(version_base=None, config_path=config_path, config_name="defaults")
def main(cfg: Config) -> None:
    """Main function for evaluation of QA datasets."""
    load_dotenv(".env")

    mlflow.openai.autolog()

    sampling_params = SamplingParams(
        temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens
    )
    runner = BatchInferenceRunner(sampling_params, cfg.model.model_name, base_url=cfg.base_url)
    sys_prompt = FileManager(cfg.dataset.sys_prompt_path).read()

    ## Run the Inference
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(experiment_name=cfg.mlflow.experiment_id)
    qa_dataset = load_dataset(cfg.dataset.name, split=cfg.dataset.split).to_pandas()
    dataset_obj = FinQADatasetCollection(
        qa_dataset, cfg.dataset.retrieval_query, cfg.dataset.meta_data_keys
    )

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
            rag_config = {
                **cfg.rag,
                "context_collection": dataset_obj.get_context_collection(),
                "collection_name": cfg.vector_db.collection_name,
                "embedding_function": cfg.vector_db.embedding_function,
                "top_k": cfg.vector_db.top_k,
                "runner": runner,
                "template_name": cfg.dataset.template_name,
            }
            rag_method_instance = RAGFactory.create(rag_config)
            responses: ResponseWrapper = dataset_obj.run(
                rag_method_instance,
                runner,
                sys_prompt,
                cfg.dataset.template_name,
                response_format=get_response_format(cfg),
            )

        json_dump = [response.to_dict() for response in responses.response_data]
        FileManager(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/inference_log.json"
        ).dump_json(json_dump)
        json_dump = [flatten_dict(response.to_dict()) for response in responses.response_data]

        active_run = mlflow.active_run()
        run_name = active_run.info.run_name if active_run else "responses"

        try:
            mlflow.log_table(data=pd.DataFrame(json_dump), artifact_file=f"{run_name}.json")
        except Exception as e:
            print(f"Failed to log table to MLflow: {e}")

        # Evaluate the retrieval
        evaluation(cfg)


if __name__ == "__main__":
    main()
