"""Unit tests for correlation analysis implementation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import CorrelationAnalyzer


class TestCorrelationAnalyzer:
    """Test CorrelationAnalyzer functionality."""

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
        analyzer = CorrelationAnalyzer()
        assert analyzer.method == "pearson"
        assert analyzer.figsize == (10, 8)

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "correlation": {
                "method": "spearman",
                "figsize": [12, 10],
                "dpi": 150,
                "colormap": "viridis",
                "output_dir": "custom_plots",
            },
            "logging": {"level": "WARNING", "file": "logs/test.log"},
        }

        config_path = self.create_temp_config(config)
        try:
            analyzer = CorrelationAnalyzer(config_path=config_path)
            assert analyzer.method == "spearman"
            assert analyzer.figsize == (12, 10)
        finally:
            Path(config_path).unlink()

    def test_load_data_from_file(self):
        """Test loading data from CSV file."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            analyzer = CorrelationAnalyzer()
            df = analyzer.load_data(file_path=csv_path)
            assert len(df) == 2
            assert len(df.columns) == 2
        finally:
            Path(csv_path).unlink()

    def test_load_data_from_dataframe(self):
        """Test loading data from pandas DataFrame."""
        df = pd.DataFrame({"age": [25, 30], "score": [85.5, 92.0]})
        analyzer = CorrelationAnalyzer()
        result = analyzer.load_data(dataframe=df)
        assert len(result) == 2

    def test_load_data_no_source(self):
        """Test that loading without source raises error."""
        analyzer = CorrelationAnalyzer()
        with pytest.raises(ValueError, match="must be provided"):
            analyzer.load_data()

    def test_get_numeric_columns(self):
        """Test getting numerical columns."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            analyzer = CorrelationAnalyzer()
            analyzer.load_data(file_path=csv_path)
            numeric_cols = analyzer.get_numeric_columns()
            assert "age" in numeric_cols
            assert "score" in numeric_cols
            assert "name" not in numeric_cols
        finally:
            Path(csv_path).unlink()

    def test_get_numeric_columns_no_data(self):
        """Test that getting columns without data raises error."""
        analyzer = CorrelationAnalyzer()
        with pytest.raises(ValueError, match="No data loaded"):
            analyzer.get_numeric_columns()

    def test_calculate_correlation(self):
        """Test correlation matrix calculation."""
        csv_content = "age,score\n25,85.5\n30,92.0\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            analyzer = CorrelationAnalyzer()
            analyzer.load_data(file_path=csv_path)
            corr_matrix = analyzer.calculate_correlation()

            assert corr_matrix is not None
            assert len(corr_matrix) == 2
            assert "age" in corr_matrix.columns
            assert "score" in corr_matrix.columns
        finally:
            Path(csv_path).unlink()

    def test_calculate_correlation_different_methods(self):
        """Test correlation calculation with different methods."""
        csv_content = "age,score\n25,85.5\n30,92.0\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            analyzer = CorrelationAnalyzer()
            analyzer.load_data(file_path=csv_path)

            pearson = analyzer.calculate_correlation(method="pearson")
            spearman = analyzer.calculate_correlation(method="spearman")
            kendall = analyzer.calculate_correlation(method="kendall")

            assert pearson is not None
            assert spearman is not None
            assert kendall is not None
        finally:
            Path(csv_path).unlink()

    def test_calculate_correlation_invalid_method(self):
        """Test that invalid correlation method raises error."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            analyzer = CorrelationAnalyzer()
            analyzer.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="Invalid method"):
                analyzer.calculate_correlation(method="invalid")
        finally:
            Path(csv_path).unlink()

    def test_get_strong_correlations(self):
        """Test getting strong correlations."""
        csv_content = "age,score,height\n25,85.5,170\n30,92.0,175\n28,88.0,172"
        csv_path = self.create_temp_csv(csv_content)

        try:
            analyzer = CorrelationAnalyzer()
            analyzer.load_data(file_path=csv_path)
            analyzer.calculate_correlation()
            strong_corr = analyzer.get_strong_correlations(threshold=0.5)

            assert isinstance(strong_corr, list)
        finally:
            Path(csv_path).unlink()

    def test_get_strong_correlations_no_matrix(self):
        """Test that getting strong correlations without matrix calculates it."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            analyzer = CorrelationAnalyzer()
            analyzer.load_data(file_path=csv_path)
            strong_corr = analyzer.get_strong_correlations()

            assert isinstance(strong_corr, list)
        finally:
            Path(csv_path).unlink()

    def test_plot_heatmap(self):
        """Test heatmap plotting."""
        csv_content = "age,score\n25,85.5\n30,92.0\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".png", delete=False
        ).name

        try:
            analyzer = CorrelationAnalyzer()
            analyzer.load_data(file_path=csv_path)
            analyzer.plot_heatmap(save_path=output_path)

            assert Path(output_path).exists()
        finally:
            Path(csv_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_plot_scatter(self):
        """Test scatter plot creation."""
        csv_content = "age,score\n25,85.5\n30,92.0\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".png", delete=False
        ).name

        try:
            analyzer = CorrelationAnalyzer()
            analyzer.load_data(file_path=csv_path)
            analyzer.plot_scatter("age", "score", save_path=output_path)

            assert Path(output_path).exists()
        finally:
            Path(csv_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_plot_scatter_invalid_columns(self):
        """Test scatter plot with invalid columns."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            analyzer = CorrelationAnalyzer()
            analyzer.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="not found"):
                analyzer.plot_scatter("invalid", "score")
        finally:
            Path(csv_path).unlink()

    def test_plot_scatter_matrix(self):
        """Test scatter plot matrix creation."""
        csv_content = "age,score,height\n25,85.5,170\n30,92.0,175\n28,88.0,172"
        csv_path = self.create_temp_csv(csv_content)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".png", delete=False
        ).name

        try:
            analyzer = CorrelationAnalyzer()
            analyzer.load_data(file_path=csv_path)
            analyzer.plot_scatter_matrix(save_path=output_path)

            assert Path(output_path).exists()
        finally:
            Path(csv_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_analyze_correlations(self):
        """Test correlation analysis."""
        csv_content = "age,score\n25,85.5\n30,92.0\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            analyzer = CorrelationAnalyzer()
            analyzer.load_data(file_path=csv_path)
            analysis = analyzer.analyze_correlations()

            assert "method" in analysis
            assert "mean_correlation" in analysis
            assert "max_correlation" in analysis
            assert "min_correlation" in analysis
            assert "strong_correlations" in analysis
        finally:
            Path(csv_path).unlink()

    def test_analyze_correlations_no_data(self):
        """Test that analysis without data raises error."""
        analyzer = CorrelationAnalyzer()
        with pytest.raises(ValueError, match="No data loaded"):
            analyzer.analyze_correlations()

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            CorrelationAnalyzer(config_path="nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
