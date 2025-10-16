"""Unit tests for load_or_cache_dataset function."""

# Mock the imports that would be available in the actual environment
import unittest
from unittest.mock import patch

from datasets import Dataset

from autoDoc.config import DatasetConfig
from autoDoc.dataset_utils import format_alpaca_example, format_grpo_example


class TestLoadAndPreprocessDataset(unittest.TestCase):
    """Test cases for load_and_preprocess_dataset function."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DatasetConfig(
            dataset_name="test/dataset",
            max_samples=10,
            train_split_ratio=0.8,
            preproc_func=None,
            eos_token="<|endoftext|>",
        )
        self.lang = "python"
        self.seed = 42

    @patch("datasets.load_dataset")
    def test_load_and_preprocess_dataset_sft(self, mock_load_dataset):
        """Test load_and_preprocess_dataset for SFT training."""
        # Mock dataset
        mock_dataset = Dataset.from_dict(
            {
                "func_code_string": ["def test(): pass", "def another(): pass"],
                "func_documentation_string": ["Test function", "Another function"],
                "language": ["python", "python"],
            }
        )
        mock_load_dataset.return_value = mock_dataset
        self.config.preproc_func = format_alpaca_example
        # Import the function
        from autoDoc.train import load_and_preprocess_dataset

        # Call the function
        train_ds, eval_ds = load_and_preprocess_dataset(
            self.config, self.lang, self.seed
        )

        # Assertions
        self.assertIsInstance(train_ds, Dataset)
        self.assertIsInstance(eval_ds, Dataset)
        self.assertIn("text", train_ds.column_names)
        self.assertIn("text", eval_ds.column_names)
        self.assertIn(self.config.eos_token, train_ds[0]["text"])
        self.assertIn(self.config.eos_token, eval_ds[0]["text"])

    @patch("datasets.load_dataset")
    def test_load_and_preprocess_dataset_grpo(self, mock_load_dataset):
        """Test load_and_preprocess_dataset for GRPO training."""
        # Update config for GRPO
        self.config.training_type = "grpo"
        self.config.preproc_func = format_grpo_example

        # Mock dataset
        mock_dataset = Dataset.from_dict(
            {
                "func_code_string": ["def test(): pass", "def another(): pass"],
                "func_documentation_string": ["Test function", "Another function"],
                "language": ["python", "python"],
            }
        )
        mock_load_dataset.return_value = mock_dataset

        # Import the function
        from autoDoc.train import load_and_preprocess_dataset

        # Call the function
        train_ds, eval_ds = load_and_preprocess_dataset(
            self.config, self.lang, self.seed
        )

        # Assertions
        self.assertIsInstance(train_ds, Dataset)
        self.assertIsInstance(eval_ds, Dataset)

        self.assertIn("prompt", train_ds.column_names)
        self.assertIn("prompt", eval_ds.column_names)
        self.assertIn("def test(): pass", train_ds[0]["prompt"])
        print(train_ds[0]["prompt"])


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
