# 🧪 RAG Eval

This repository compares different RAG methods. It uses following tools and libraries to make it easier to run experiments and compare results:

- 📦 **[uv](https://github.com/astral-sh/uv)**
  - A Python package manager replacing `pip` and `poetry`.
- 📝 **[pyproject](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/)**
  - For project management.
- ⚙️ **[hydra](https://hydra.cc)**
  - For flexible configuration management.
- 📊 **[wandb](https://wandb.ai/site)**
  - For experiment tracking and visualization online.
- 🤖 **encourage**
  - A custom library for handling LLM inference, prompt handling, and utility functions.

---

## 🚀 Initialization

### 📦 UV

To initialize the environment using `uv`, run the following command:

```bash
uv sync
```

### 📊 Weights & Biases (wandb)

For using Weight & Biases you need to login to your account using the following command:

```bash
wandb login
```

There you have to enter your API key which you can find in your account settings.

## ⚡ Usage

When using this template you have to declare all your configuration parameters in the `conf/defaults.yaml` file. Also modify the `conf/model/defaults.yaml` and `conf/data/defaults.yaml` files to fit your needs.

To run an experiment you can use the following command. The parameter follow the hydra syntax.

```bash
uv run main.py +experiment=experiment_name
```

The output will be saved in the outputs folder. Each experiment generates a new timestamped folder containing:

- Configuration files (.hydra/)
- Logs (main.log)
- Inference output (inference_log.json)
