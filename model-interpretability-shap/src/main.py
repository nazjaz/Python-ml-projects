"""Model Interpretability using SHAP Values, Permutation Importance, and Partial Dependence.

This module provides implementations of model interpretability techniques including
SHAP (SHapley Additive exPlanations) values, permutation importance, and partial
dependence plots for understanding model predictions and feature importance.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("SHAP not available. Install with: pip install shap")

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP (SHapley Additive exPlanations) values for model interpretability."""

    def __init__(
        self,
        model,
        X_background: Optional[np.ndarray] = None,
        algorithm: str = "auto",
        max_evals: int = 100,
    ):
        """Initialize SHAP explainer.

        Args:
            model: Trained model (must have predict or predict_proba method)
            X_background: Background dataset for SHAP (default: None, uses subset)
            algorithm: SHAP algorithm - "auto", "exact", "tree", "linear", "kernel"
                (default: "auto")
            max_evals: Maximum evaluations for kernel SHAP (default: 100)
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            )

        self.model = model
        self.X_background = X_background
        self.algorithm = algorithm
        self.max_evals = max_evals
        self.explainer = None
        self.shap_values_ = None

    def fit(self, X: np.ndarray) -> "SHAPExplainer":
        """Fit SHAP explainer to data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If input data is invalid
        """
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if self.X_background is None:
            if len(X) > 100:
                np.random.seed(42)
                indices = np.random.choice(len(X), 100, replace=False)
                self.X_background = X[indices]
            else:
                self.X_background = X

        try:
            if self.algorithm == "auto":
                if hasattr(self.model, "predict_proba"):
                    try:
                        self.explainer = shap.TreeExplainer(self.model)
                        self.algorithm = "tree"
                    except Exception:
                        try:
                            self.explainer = shap.LinearExplainer(
                                self.model, self.X_background
                            )
                            self.algorithm = "linear"
                        except Exception:
                            self.explainer = shap.KernelExplainer(
                                self.model.predict, self.X_background
                            )
                            self.algorithm = "kernel"
                else:
                    try:
                        self.explainer = shap.TreeExplainer(self.model)
                        self.algorithm = "tree"
                    except Exception:
                        self.explainer = shap.KernelExplainer(
                            self.model.predict, self.X_background
                        )
                        self.algorithm = "kernel"
            elif self.algorithm == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.algorithm == "linear":
                self.explainer = shap.LinearExplainer(self.model, self.X_background)
            elif self.algorithm == "kernel":
                self.explainer = shap.KernelExplainer(
                    self.model.predict, self.X_background
                )
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            logger.info(f"SHAP explainer initialized with algorithm: {self.algorithm}")

        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise ValueError(f"SHAP explainer initialization failed: {e}") from e

        return self

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Calculate SHAP values for data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            SHAP values, shape (n_samples, n_features) or (n_samples, n_features, n_classes)

        Raises:
            ValueError: If explainer is not fitted
        """
        if self.explainer is None:
            raise ValueError("Explainer must be fitted before explaining")

        X = np.array(X)

        try:
            if self.algorithm == "kernel":
                shap_values = self.explainer.shap_values(
                    X, nsamples=self.max_evals, silent=True
                )
            else:
                shap_values = self.explainer.shap_values(X)

            self.shap_values_ = shap_values

            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)

            return shap_values
        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {e}")
            raise ValueError(f"SHAP value calculation failed: {e}") from e

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get feature importance from mean absolute SHAP values.

        Args:
            feature_names: Optional list of feature names (default: None)

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            ValueError: If SHAP values not calculated
        """
        if self.shap_values_ is None:
            raise ValueError("SHAP values must be calculated first")

        shap_values = self.shap_values_

        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                shap_values = np.abs(shap_values).mean(axis=0).mean(axis=0)
            else:
                shap_values = np.abs(shap_values).mean(axis=0)
        else:
            if shap_values.ndim == 3:
                shap_values = np.abs(shap_values).mean(axis=0).mean(axis=0)
            elif shap_values.ndim == 2:
                shap_values = np.abs(shap_values).mean(axis=0)
            else:
                shap_values = np.abs(shap_values)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(shap_values))]

        return {
            name: float(score) for name, score in zip(feature_names, shap_values)
        }

    def plot_summary(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_display: int = 10,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot SHAP summary plot.

        Args:
            X: Feature matrix
            feature_names: Optional list of feature names
            max_display: Maximum number of features to display (default: 10)
            save_path: Path to save figure (default: None)
        """
        if self.explainer is None:
            raise ValueError("Explainer must be fitted before plotting")

        shap_values = self.explain(X)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        try:
            shap.summary_plot(
                shap_values,
                X,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
            )

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"SHAP summary plot saved to {save_path}")

            plt.close()
        except Exception as e:
            logger.error(f"Failed to create SHAP summary plot: {e}")
            raise


class PermutationImportanceCalculator:
    """Calculate permutation importance for model interpretability."""

    def __init__(
        self,
        model,
        scoring: Optional[Union[str, callable]] = None,
        n_repeats: int = 5,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        """Initialize permutation importance calculator.

        Args:
            model: Trained model
            scoring: Scoring function or string (default: None, auto-selects)
            n_repeats: Number of times to permute each feature (default: 5)
            random_state: Random seed (default: None)
            n_jobs: Number of parallel jobs (default: None)
        """
        self.model = model
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_ = None
        self.feature_names_ = None

    def calculate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate permutation importance.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target vector, shape (n_samples,)
            feature_names: Optional list of feature names (default: None)

        Returns:
            Dictionary with importance statistics for each feature

        Raises:
            ValueError: If input data is invalid
        """
        X = np.array(X)
        y = np.array(y).ravel()

        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X.shape[0]={X.shape[0]}, len(y)={len(y)}"
            )

        if self.scoring is None:
            if hasattr(self.model, "predict_proba"):
                self.scoring = "accuracy"
            else:
                self.scoring = "r2"

        result = permutation_importance(
            self.model,
            X,
            y,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.importance_ = result
        self.feature_names_ = (
            feature_names if feature_names else [f"feature_{i}" for i in range(X.shape[1])]
        )

        importance_dict = {}
        for i, name in enumerate(self.feature_names_):
            importance_dict[name] = {
                "importance_mean": float(result.importances_mean[i]),
                "importance_std": float(result.importances_std[i]),
            }

        logger.info(f"Permutation importance calculated for {len(self.feature_names_)} features")

        return importance_dict

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            ValueError: If importance not calculated
        """
        if self.importance_ is None:
            raise ValueError("Permutation importance must be calculated first")

        return {
            name: float(self.importance_.importances_mean[i])
            for i, name in enumerate(self.feature_names_)
        }

    def plot_importance(
        self,
        top_n: int = 10,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """Plot permutation importance.

        Args:
            top_n: Number of top features to display (default: 10)
            save_path: Path to save figure (default: None)
            figsize: Figure size (default: (10, 6))
        """
        if self.importance_ is None:
            raise ValueError("Permutation importance must be calculated first")

        importances_mean = self.importance_.importances_mean
        importances_std = self.importance_.importances_std

        top_indices = np.argsort(importances_mean)[-top_n:][::-1]

        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(len(top_indices))
        ax.barh(
            y_pos,
            importances_mean[top_indices],
            xerr=importances_std[top_indices],
            align="center",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names_[i] for i in top_indices])
        ax.set_xlabel("Permutation Importance")
        ax.set_title(f"Top {top_n} Features by Permutation Importance")
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Permutation importance plot saved to {save_path}")

        plt.close()


class PartialDependencePlotter:
    """Generate partial dependence plots for model interpretability."""

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """Initialize partial dependence plotter.

        Args:
            model: Trained model
            feature_names: Optional list of feature names (default: None)
        """
        self.model = model
        self.feature_names = feature_names

    def plot(
        self,
        X: np.ndarray,
        features: Union[int, str, List[Union[int, str]]],
        feature_names: Optional[List[str]] = None,
        grid_resolution: int = 100,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> None:
        """Generate partial dependence plot.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            features: Feature(s) to plot - index, name, or list
            feature_names: Optional list of feature names (default: None)
            grid_resolution: Number of grid points (default: 100)
            save_path: Path to save figure (default: None)
            figsize: Figure size (default: (12, 6))
        """
        X = np.array(X)

        if feature_names is None:
            feature_names = (
                self.feature_names
                if self.feature_names
                else [f"feature_{i}" for i in range(X.shape[1])]
            )

        if isinstance(features, (int, str)):
            features = [features]

        feature_indices = []
        for feat in features:
            if isinstance(feat, str):
                if feat in feature_names:
                    feature_indices.append(feature_names.index(feat))
                else:
                    raise ValueError(f"Feature '{feat}' not found in feature_names")
            else:
                feature_indices.append(int(feat))

        try:
            display = PartialDependenceDisplay.from_estimator(
                self.model,
                X,
                features=feature_indices,
                feature_names=feature_names,
                grid_resolution=grid_resolution,
            )

            display.figure_.set_size_inches(figsize)
            display.figure_.tight_layout()

            if save_path:
                display.figure_.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Partial dependence plot saved to {save_path}")

            plt.close(display.figure_)
        except Exception as e:
            logger.error(f"Failed to create partial dependence plot: {e}")
            raise


class ModelInterpreter:
    """Main model interpretability class combining all techniques."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize model interpreter.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.model = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.shap_explainer = None
        self.perm_importance = None
        self.pdp_plotter = None

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "logs/app.log")

        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(handler)

    def load_model_and_data(
        self,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "ModelInterpreter":
        """Load model and data for interpretation.

        Args:
            model: Trained model
            X: Feature matrix
            y: Optional target vector (needed for permutation importance)

        Returns:
            Self for method chaining
        """
        self.model = model

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            X = np.array(X)

        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            else:
                y = np.array(y)

        self.X = X
        self.y = y

        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        logger.info(
            f"Loaded model and data: {X.shape[0]} samples, {X.shape[1]} features"
        )

        return self

    def calculate_shap(
        self, X: Optional[np.ndarray] = None, algorithm: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate SHAP values and feature importance.

        Args:
            X: Feature matrix (default: None, uses loaded data)
            algorithm: SHAP algorithm (default: None, auto-selects)

        Returns:
            Dictionary with feature importance from SHAP

        Raises:
            ValueError: If model or data not loaded
        """
        if self.model is None or self.X is None:
            raise ValueError("Model and data must be loaded first")

        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")

        X_use = X if X is not None else self.X

        shap_config = self.config.get("shap", {})
        algorithm = algorithm or shap_config.get("algorithm", "auto")

        self.shap_explainer = SHAPExplainer(
            self.model, algorithm=algorithm, max_evals=shap_config.get("max_evals", 100)
        )
        self.shap_explainer.fit(X_use)

        shap_values = self.shap_explainer.explain(X_use)
        importance = self.shap_explainer.get_feature_importance(self.feature_names)

        return importance

    def calculate_permutation_importance(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        scoring: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate permutation importance.

        Args:
            X: Feature matrix (default: None, uses loaded data)
            y: Target vector (default: None, uses loaded data)
            scoring: Scoring metric (default: None, auto-selects)

        Returns:
            Dictionary with importance statistics

        Raises:
            ValueError: If model or data not loaded
        """
        if self.model is None or self.X is None or self.y is None:
            raise ValueError("Model, X, and y must be loaded first")

        X_use = X if X is not None else self.X
        y_use = y if y is not None else self.y

        perm_config = self.config.get("permutation_importance", {})
        n_repeats = perm_config.get("n_repeats", 5)

        self.perm_importance = PermutationImportanceCalculator(
            self.model,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=perm_config.get("random_state", None),
        )

        importance = self.perm_importance.calculate(
            X_use, y_use, feature_names=self.feature_names
        )

        return importance

    def plot_partial_dependence(
        self,
        features: Union[int, str, List[Union[int, str]]],
        X: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """Generate partial dependence plot.

        Args:
            features: Feature(s) to plot
            X: Feature matrix (default: None, uses loaded data)
            save_path: Path to save figure (default: None)

        Raises:
            ValueError: If model or data not loaded
        """
        if self.model is None or self.X is None:
            raise ValueError("Model and data must be loaded first")

        X_use = X if X is not None else self.X

        if self.pdp_plotter is None:
            self.pdp_plotter = PartialDependencePlotter(
                self.model, feature_names=self.feature_names
            )

        self.pdp_plotter.plot(
            X_use, features, feature_names=self.feature_names, save_path=save_path
        )

    def generate_all_interpretations(
        self, output_dir: Optional[Path] = None
    ) -> Dict:
        """Generate all interpretability analyses.

        Args:
            output_dir: Directory to save plots (default: None)

        Returns:
            Dictionary with all interpretation results

        Raises:
            ValueError: If model or data not loaded
        """
        if self.model is None or self.X is None:
            raise ValueError("Model and data must be loaded first")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        if SHAP_AVAILABLE:
            try:
                shap_importance = self.calculate_shap()
                results["shap"] = shap_importance

                if output_dir:
                    self.shap_explainer.plot_summary(
                        self.X,
                        feature_names=self.feature_names,
                        save_path=output_dir / "shap_summary.png",
                    )
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")
                results["shap"] = {"error": str(e)}

        if self.y is not None:
            try:
                perm_importance = self.calculate_permutation_importance()
                results["permutation_importance"] = perm_importance

                if output_dir and self.perm_importance:
                    self.perm_importance.plot_importance(
                        save_path=output_dir / "permutation_importance.png"
                    )
            except Exception as e:
                logger.warning(f"Permutation importance calculation failed: {e}")
                results["permutation_importance"] = {"error": str(e)}

        if output_dir:
            top_features = list(results.get("shap", {}).keys())[:3]
            if not top_features and results.get("permutation_importance"):
                top_features = sorted(
                    results["permutation_importance"].keys(),
                    key=lambda x: results["permutation_importance"][x]["importance_mean"],
                    reverse=True,
                )[:3]

            if top_features:
                try:
                    self.plot_partial_dependence(
                        top_features[0],
                        save_path=output_dir / "partial_dependence.png",
                    )
                except Exception as e:
                    logger.warning(f"Partial dependence plot failed: {e}")

        return results


def main():
    """Main entry point for model interpreter."""
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
        description="Model interpretability using SHAP, permutation importance, and PDP"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to pickled model file",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV file with features",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        help="Column name for target variable (optional, for permutation importance)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save interpretation plots and results",
    )
    parser.add_argument(
        "--results-output",
        type=str,
        help="Path to output JSON file for interpretation results",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    interpreter = ModelInterpreter(
        config_path=Path(args.config) if args.config else None
    )

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(args.data)

    if args.target_col and args.target_col in df.columns:
        X = df.drop(args.target_col, axis=1)
        y = df[args.target_col]
    else:
        X = df
        y = None

    interpreter.load_model_and_data(model, X, y)

    output_dir = Path(args.output_dir) if args.output_dir else None

    results = interpreter.generate_all_interpretations(output_dir=output_dir)

    if args.results_output:
        results_export = {}
        for method, data in results.items():
            if "error" not in data:
                if method == "shap":
                    results_export[method] = data
                elif method == "permutation_importance":
                    results_export[method] = {
                        k: v["importance_mean"] for k, v in data.items()
                    }

        with open(args.results_output, "w") as f:
            json.dump(results_export, f, indent=2)
        logger.info(f"Interpretation results saved to {args.results_output}")

    print("\nModel Interpretation Results:")
    print("=" * 50)

    if "shap" in results:
        if "error" not in results["shap"]:
            print("\nSHAP Feature Importance (Top 10):")
            sorted_shap = sorted(
                results["shap"].items(), key=lambda x: x[1], reverse=True
            )[:10]
            for feature, importance in sorted_shap:
                print(f"  {feature}: {importance:.4f}")
        else:
            print(f"\nSHAP Error: {results['shap']['error']}")

    if "permutation_importance" in results:
        if "error" not in results["permutation_importance"]:
            print("\nPermutation Importance (Top 10):")
            sorted_perm = sorted(
                results["permutation_importance"].items(),
                key=lambda x: x[1]["importance_mean"],
                reverse=True,
            )[:10]
            for feature, stats in sorted_perm:
                print(
                    f"  {feature}: {stats['importance_mean']:.4f} "
                    f"(±{stats['importance_std']:.4f})"
                )
        else:
            print(f"\nPermutation Importance Error: {results['permutation_importance']['error']}")


if __name__ == "__main__":
    main()
