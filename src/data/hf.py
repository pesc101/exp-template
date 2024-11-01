"""HuggingFaceDataset class for loading datasets from HuggingFace datasets library."""

from datasets import load_dataset


class HuggingFaceDataset:
    """HuggingFaceDataset class for loading datasets from HuggingFace datasets library."""

    def __init__(
        self, dataset_path: str, dataset_name: str, split: str, max_samples: int = 0
    ) -> None:
        self.raw_data = load_dataset(dataset_path, dataset_name, split=split).to_pandas()
        if max_samples > 0:
            self.raw_data = self.raw_data.head(max_samples)
        self.reformat_raw_data()

    def reformat_raw_data(self):
        """Reformat raw data to be more accessible."""
        self.messages = self.from_col_to_list("messages")
        self.answers = self.from_col_to_list("answers")
        self.ctxs = self.from_col_to_list("ctxs")
        self.ground_truth_ctx = self.from_col_to_list("ground_truth_ctx")

    def from_col_to_list(self, col_name: str) -> list[str]:
        """Convert pandas column to list."""
        if col_name not in self.raw_data.columns:
            return []
        return self.raw_data[col_name].tolist()

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.raw_data)
