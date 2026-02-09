"""Unit tests for data visualization implementation."""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import DataVisualizer


class TestDataVisualizer:
    """Test DataVisualizer functionality."""

    def create_temp_csv(self, content: str) -> str:
        """Create temporary CSV file for testing.

        Args:
            content: CSV content as string.

        Returns:
            Path to temporary CSV file.
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        temp_file.write(content)
        temp_file.close()
        return temp_file.name

    def create_temp_config(self, config_dict: dict) -> str:
        """Create temporary config file for testing.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Path to temporary config file.
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name

    def test_initialization_with_default_config(self):
        """Test initialization with default config file."""
        visualizer = DataVisualizer()
        assert visualizer.figsize == (10, 6)
        assert visualizer.dpi == 100

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "visualization": {
                "figsize": [12, 8],
                "dpi": 150,
                "style": "darkgrid",
                "color_palette": "Set2",
                "output_dir": "custom_plots",
            },
            "logging": {"level": "WARNING", "file": "logs/test.log"},
        }

        config_path = self.create_temp_config(config)
        try:
            visualizer = DataVisualizer(config_path=config_path)
            assert visualizer.figsize == (12, 8)
            assert visualizer.dpi == 150
        finally:
            Path(config_path).unlink()

    def test_load_data_from_file(self):
        """Test loading data from CSV file."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            visualizer = DataVisualizer()
            df = visualizer.load_data(file_path=csv_path)
            assert len(df) == 2
            assert len(df.columns) == 2
        finally:
            Path(csv_path).unlink()

    def test_load_data_from_dataframe(self):
        """Test loading data from pandas DataFrame."""
        df = pd.DataFrame({"age": [25, 30], "score": [85.5, 92.0]})
        visualizer = DataVisualizer()
        result = visualizer.load_data(dataframe=df)
        assert len(result) == 2

    def test_load_data_no_source(self):
        """Test that loading without source raises error."""
        visualizer = DataVisualizer()
        with pytest.raises(ValueError, match="must be provided"):
            visualizer.load_data()

    def test_get_numeric_columns(self):
        """Test getting numerical columns."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            visualizer = DataVisualizer()
            visualizer.load_data(file_path=csv_path)
            numeric_cols = visualizer.get_numeric_columns()
            assert "age" in numeric_cols
            assert "score" in numeric_cols
            assert "name" not in numeric_cols
        finally:
            Path(csv_path).unlink()

    def test_get_numeric_columns_no_data(self):
        """Test that getting columns without data raises error."""
        visualizer = DataVisualizer()
        with pytest.raises(ValueError, match="No data loaded"):
            visualizer.get_numeric_columns()

    def test_plot_histogram(self):
        """Test histogram plotting."""
        csv_content = "age,score\n25,85.5\n30,92.0\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".png", delete=False
        ).name

        try:
            visualizer = DataVisualizer()
            visualizer.load_data(file_path=csv_path)
            visualizer.plot_histogram(save_path=output_path)

            assert Path(output_path).exists()
        finally:
            Path(csv_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_plot_histogram_specific_columns(self):
        """Test histogram plotting for specific columns."""
        csv_content = "age,score,height\n25,85.5,170\n30,92.0,175"
        csv_path = self.create_temp_csv(csv_content)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".png", delete=False
        ).name

        try:
            visualizer = DataVisualizer()
            visualizer.load_data(file_path=csv_path)
            visualizer.plot_histogram(columns=["age"], save_path=output_path)

            assert Path(output_path).exists()
        finally:
            Path(csv_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_plot_boxplot(self):
        """Test box plot plotting."""
        csv_content = "age,score\n25,85.5\n30,92.0\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".png", delete=False
        ).name

        try:
            visualizer = DataVisualizer()
            visualizer.load_data(file_path=csv_path)
            visualizer.plot_boxplot(save_path=output_path)

            assert Path(output_path).exists()
        finally:
            Path(csv_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_plot_density(self):
        """Test density plot plotting."""
        csv_content = "age,score\n25,85.5\n30,92.0\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".png", delete=False
        ).name

        try:
            visualizer = DataVisualizer()
            visualizer.load_data(file_path=csv_path)
            visualizer.plot_density(save_path=output_path)

            assert Path(output_path).exists()
        finally:
            Path(csv_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_plot_all_distributions(self):
        """Test plotting all distribution types."""
        csv_content = "age,score\n25,85.5\n30,92.0\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)
        output_dir = tempfile.mkdtemp()

        try:
            visualizer = DataVisualizer()
            visualizer.load_data(file_path=csv_path)
            visualizer.plot_all_distributions(save_dir=output_dir)

            assert Path(f"{output_dir}/histograms.png").exists()
            assert Path(f"{output_dir}/boxplots.png").exists()
            assert Path(f"{output_dir}/density_plots.png").exists()
        finally:
            Path(csv_path).unlink()
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_plot_histogram_no_numeric(self):
        """Test histogram with no numerical columns."""
        csv_content = "name,category\nAlice,TypeA\nBob,TypeB"
        csv_path = self.create_temp_csv(csv_content)

        try:
            visualizer = DataVisualizer()
            visualizer.load_data(file_path=csv_path)
            visualizer.plot_histogram()
        finally:
            Path(csv_path).unlink()

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            DataVisualizer(config_path="nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
