"""FinQA dataset model."""

import json
import uuid
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts.context import Document
from encourage.prompts.meta_data import MetaData
from encourage.rag import RAGMethodInterface
from pydantic import BaseModel


class FinQADatasetSample(BaseModel):
    """FinQA dataset model."""

    id: str
    question: Optional[str] = None
    answer: Optional[str] = None
    program_solution: Optional[str] = None
    pre_text: Optional[str] = None
    post_text: Optional[str] = None
    table: Optional[str] = None
    context: Optional[str] = None
    report_year: Optional[int] = None
    page_number: Optional[int] = None
    company_symbol: Optional[str] = None
    company_name: Optional[str] = None
    company_sector: Optional[str] = None
    company_industry: Optional[str] = None
    company_headquarters: Optional[str] = None
    company_date_added: Optional[str] = None
    company_cik: Optional[int] = None
    company_founded: Optional[str] = None
    question_de: Optional[str] = None
    context_de: Optional[str] = None
    program_answer: Optional[str] = None
    context_id: Optional[str] = None


class FinQADatasetCollection:
    """Collection of FinQA dataset samples."""

    def __init__(
        self,
        df: pd.DataFrame,
        retrieval_query: str = "",
        meta_data_keys: list[str] = [],
        document_percentage: float | None = None,
    ) -> None:
        """Initialize the dataset collection."""
        self.meta_data_keys = meta_data_keys
        self.retrieval_query = retrieval_query

        # Create samples from DataFrame records
        # Filter DataFrame to only keep required columns
        required_columns = [
            "id",
            "question",
            "answer",
            "program_answer",
            "program_solution",
            "context",
        ] + self.meta_data_keys
        df = df[[col for col in required_columns if col in df.columns]]
        # Ensure 'context' column is a single string joined by newlines if it's a list/tuple
        if "context" in df.columns:

            def _join_context(val):
                if isinstance(val, np.ndarray):
                    return "\n\n".join(map(str, val.flatten()))
                return val

            df.loc[:, "context"] = df["context"].apply(_join_context)
        self.samples = [
            FinQADatasetSample(**{str(k): v for k, v in record.items()})  # ty: ignore
            for record in df.to_dict(orient="records")
        ]

        # Create context IDs for each sample
        self.create_context_ids()

        if document_percentage and document_percentage < 1.0:
            # Sample a percentage of unique context IDs in the DataFrame
            unique_context_ids = pd.Series(
                [sample.context_id for sample in self.samples]
            ).drop_duplicates()
            sampled_context_ids = unique_context_ids.sample(frac=document_percentage)

            mlflow.log_params({"unique_documents": len(sampled_context_ids)})  # ty: ignore
            self.samples = [
                sample
                for sample in self.samples
                if sample.context_id in sampled_context_ids.tolist()
            ]

        # Create metadata and prepare user prompts using samples
        self.prompt_meta_data = self.create_prompt_meta_data()
        self.user_prompts = [sample.question for sample in self.samples]
        self.context_collection = self.prepare_contexts_for_db()

    def get_context_collection(self) -> list[Document]:
        """Get the context collection."""
        return self.context_collection

    def get_data_frame(self) -> pd.DataFrame:
        """Get the DataFrame."""
        return pd.DataFrame([sample.dict() for sample in self.samples])

    def create_context_ids(self) -> None:
        """Create context ID for each sample and update both samples and qa_dataset."""
        # Create mapping of unique contexts to UUIDs
        context_dict = {}
        for sample in self.samples:
            if sample.context not in context_dict:
                context_dict[sample.context] = str(uuid.uuid4())
            sample.context_id = context_dict[sample.context]

    def create_prompt_meta_data(self) -> list[MetaData]:
        """Create metadata from samples."""
        meta_datas = []
        for _, sample in enumerate(self.samples):
            meta_data = MetaData(
                {
                    "reference_answer": sample.program_answer or "",
                    "id": sample.id,
                    "reference_document": Document(
                        id=uuid.UUID(sample.context_id) if sample.context_id else uuid.uuid4(),
                        content=sample.context or "",
                    ),
                }
            )
            meta_datas.append(meta_data)
        return meta_datas

    def prepare_contexts_for_db(self) -> list[Document]:
        """Prepare contexts from samples for the vector database."""
        context_collection = []
        for sample in self.samples:
            context = Document(
                id=uuid.UUID(sample.context_id)
                if isinstance(sample.context_id, str)
                else sample.context_id or uuid.uuid4(),
                content=sample.context or "",
                meta_data=MetaData(
                    {key: getattr(sample, key, None) for key in self.meta_data_keys}
                ),
            )
            context_collection.append(context)
        return context_collection

    def run(
        self,
        rag_method_instance: RAGMethodInterface,
        runner: BatchInferenceRunner,
        sys_prompt: dict,
        template_name: str = "",
        response_format: type[BaseModel] | str | None = None,
    ) -> ResponseWrapper:
        """Run the dataset."""
        retrieval_query = self._generate_retrieval_queries()
        responses = rag_method_instance.run(
            runner,
            sys_prompt["round1"],
            self.user_prompts,
            self.prompt_meta_data,
            retrieval_queries=retrieval_query,
            response_format=response_format,
        )
        return self.post_response_processing(responses)

    def post_response_processing(
        self,
        responses: ResponseWrapper,
    ) -> ResponseWrapper:
        """Post-process the response."""
        for response in responses.response_data:
            try:
                json_response = json.loads(response.response)
                response.response = json_response["computed_formula"]
                response.meta_data["reasoning_steps"] = json_response["reasoning_steps"]
                response.meta_data["final_formula"] = json_response["final_formula"]
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Response: {response.response}")
                response.response = ""
            except TypeError as e:
                print(f"TypeError: {e}")
                print(f"Response: {response.response}")
                response.response = ""
        return responses

    def _generate_retrieval_queries(self) -> list[str]:
        """Generate queries from samples."""
        queries = [f"{sample.company_name} : {sample.question}" for sample in self.samples]
        if self.retrieval_query:
            queries = [f"Instruct: {self.retrieval_query}\nQuery: {q}" for q in queries]
        return queries
