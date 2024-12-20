from dataclasses import dataclass


@dataclass
class Model:
    model_name: str
    model_name_short: str
    gpu_memory_utilization: float
    temperature: float
    max_tokens: int
    top_p: float = 0.95
    seed: int | None = None


@dataclass
class Dataset:
    name: str


@dataclass
class WandB:
    project: str
    entity: str
    run_name: str


@dataclass
class Config:
    model: Model
    datset: Dataset
    wandb: WandB
    sys_prompt_path: str
    tools_file_path: str
    output_file_path: str
    template_name: str = "generate_python_code_conv.j2"
