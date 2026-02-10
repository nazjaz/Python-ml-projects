"""Model Persistence and Versioning with Serialization and Metadata Tracking.

This module provides comprehensive model persistence, versioning, and metadata
tracking capabilities for machine learning models, including serialization,
loading, version management, and model registry functionality.
"""

import hashlib
import json
import logging
import logging.handlers
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.base import BaseEstimator

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ModelMetadata:
    """Model metadata container."""

    def __init__(
        self,
        model_name: str,
        model_type: str,
        version: str,
        created_at: Optional[str] = None,
        **kwargs,
    ):
        """Initialize model metadata.

        Args:
            model_name: Name of the model
            model_type: Type of model (e.g., 'RandomForest', 'LinearRegression')
            version: Version string (e.g., '1.0.0', 'v1.2.3')
            created_at: Creation timestamp (default: current time)
            **kwargs: Additional metadata fields
        """
        self.model_name = model_name
        self.model_type = model_type
        self.version = version
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = self.created_at
        self.metadata = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary.

        Returns:
            Dictionary representation of metadata
        """
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary.

        Args:
            data: Dictionary containing metadata

        Returns:
            ModelMetadata instance
        """
        required_fields = ["model_name", "model_type", "version"]
        metadata_fields = {k: v for k, v in data.items() if k not in required_fields}

        return cls(
            model_name=data["model_name"],
            model_type=data["model_type"],
            version=data["version"],
            created_at=data.get("created_at"),
            **metadata_fields,
        )

    def update(self, **kwargs) -> None:
        """Update metadata fields.

        Args:
            **kwargs: Fields to update
        """
        self.metadata.update(kwargs)
        self.updated_at = datetime.now().isoformat()


class ModelVersion:
    """Model version information."""

    def __init__(
        self,
        version: str,
        model_path: Path,
        metadata_path: Path,
        metadata: ModelMetadata,
    ):
        """Initialize model version.

        Args:
            version: Version string
            model_path: Path to serialized model file
            metadata_path: Path to metadata file
            metadata: ModelMetadata instance
        """
        self.version = version
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary.

        Returns:
            Dictionary representation of version
        """
        return {
            "version": self.version,
            "model_path": str(self.model_path),
            "metadata_path": str(self.metadata_path),
            "metadata": self.metadata.to_dict(),
        }


class ModelSerializer:
    """Model serialization handler."""

    def __init__(self, serialization_format: str = "joblib"):
        """Initialize serializer.

        Args:
            serialization_format: Format to use - "pickle" or "joblib" (default: "joblib")
        """
        self.serialization_format = serialization_format.lower()

        if self.serialization_format not in ["pickle", "joblib"]:
            raise ValueError(
                f"Unsupported serialization format: {serialization_format}. "
                "Use 'pickle' or 'joblib'"
            )

    def serialize(
        self, model: BaseEstimator, filepath: Path, compress: bool = True
    ) -> None:
        """Serialize model to file.

        Args:
            model: Model to serialize
            filepath: Path to save model
            compress: Whether to use compression (default: True)

        Raises:
            ValueError: If model is not serializable
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            if self.serialization_format == "joblib":
                if compress:
                    joblib.dump(model, filepath, compress=("gzip", 3))
                else:
                    joblib.dump(model, filepath)
            else:  # pickle
                mode = "wb"
                if compress:
                    import gzip

                    with gzip.open(filepath.with_suffix(".pkl.gz"), mode) as f:
                        pickle.dump(model, f)
                else:
                    with open(filepath, mode) as f:
                        pickle.dump(model, f)

            logger.info(f"Model serialized to {filepath}")
        except Exception as e:
            logger.error(f"Failed to serialize model: {e}")
            raise ValueError(f"Model serialization failed: {e}") from e

    def deserialize(self, filepath: Path) -> BaseEstimator:
        """Deserialize model from file.

        Args:
            filepath: Path to serialized model file

        Returns:
            Deserialized model

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If deserialization fails
        """
        if not filepath.exists():
            # Try compressed pickle
            compressed_path = filepath.with_suffix(".pkl.gz")
            if compressed_path.exists():
                filepath = compressed_path
            else:
                raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            if self.serialization_format == "joblib" or filepath.suffix == ".joblib":
                model = joblib.load(filepath)
            else:
                if filepath.suffix == ".gz":
                    import gzip

                    with gzip.open(filepath, "rb") as f:
                        model = pickle.load(f)
                else:
                    with open(filepath, "rb") as f:
                        model = pickle.load(f)

            logger.info(f"Model deserialized from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Failed to deserialize model: {e}")
            raise ValueError(f"Model deserialization failed: {e}") from e


class ModelRegistry:
    """Model registry for version management."""

    def __init__(self, registry_path: Path):
        """Initialize model registry.

        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load registry from file.

        Returns:
            Dictionary containing registry data
        """
        if not self.registry_file.exists():
            return {}

        try:
            with open(self.registry_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return {}

    def _save_registry(self, registry: Dict[str, Dict[str, Any]]) -> None:
        """Save registry to file.

        Args:
            registry: Registry data to save
        """
        try:
            with open(self.registry_file, "w") as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise

    def register(
        self,
        model_name: str,
        version: str,
        model_path: Path,
        metadata: ModelMetadata,
    ) -> None:
        """Register model version in registry.

        Args:
            model_name: Name of the model
            version: Version string
            model_path: Path to model file
            metadata: Model metadata
        """
        registry = self._load_registry()

        if model_name not in registry:
            registry[model_name] = {}

        registry[model_name][version] = {
            "model_path": str(model_path),
            "metadata_path": str(model_path.parent / f"{model_path.stem}_metadata.json"),
            "metadata": metadata.to_dict(),
            "registered_at": datetime.now().isoformat(),
        }

        self._save_registry(registry)
        logger.info(f"Registered {model_name} version {version}")

    def get_versions(self, model_name: str) -> List[str]:
        """Get all versions for a model.

        Args:
            model_name: Name of the model

        Returns:
            List of version strings
        """
        registry = self._load_registry()
        if model_name not in registry:
            return []
        return sorted(registry[model_name].keys(), reverse=True)

    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get latest version for a model.

        Args:
            model_name: Name of the model

        Returns:
            Latest version string or None
        """
        versions = self.get_versions(model_name)
        return versions[0] if versions else None

    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get model information from registry.

        Args:
            model_name: Name of the model
            version: Version string (default: latest)

        Returns:
            Model information dictionary or None
        """
        registry = self._load_registry()
        if model_name not in registry:
            return None

        if version is None:
            version = self.get_latest_version(model_name)
            if version is None:
                return None

        if version not in registry[model_name]:
            return None

        return registry[model_name][version]

    def list_models(self) -> List[str]:
        """List all registered models.

        Returns:
            List of model names
        """
        registry = self._load_registry()
        return sorted(registry.keys())


class ModelPersistence:
    """Main model persistence and versioning manager."""

    def __init__(
        self,
        models_dir: Union[str, Path] = "models",
        registry_path: Optional[Union[str, Path]] = None,
        serialization_format: str = "joblib",
        compress: bool = True,
    ):
        """Initialize model persistence manager.

        Args:
            models_dir: Directory to store models (default: "models")
            registry_path: Path to registry directory (default: models_dir)
            serialization_format: Serialization format (default: "joblib")
            compress: Whether to compress models (default: True)
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if registry_path is None:
            registry_path = self.models_dir
        self.registry = ModelRegistry(registry_path)

        self.serializer = ModelSerializer(serialization_format)
        self.compress = compress

    def save(
        self,
        model: BaseEstimator,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model with versioning and metadata.

        Args:
            model: Model to save
            model_name: Name of the model
            version: Version string
            metadata: Base metadata dictionary (optional)
            additional_metadata: Additional metadata to include (optional)

        Returns:
            Path to saved model file

        Raises:
            ValueError: If model or parameters are invalid
        """
        if not isinstance(model, BaseEstimator):
            raise ValueError("Model must be a scikit-learn BaseEstimator")

        model_type = type(model).__name__

        # Create metadata
        model_metadata = ModelMetadata(
            model_name=model_name,
            model_type=model_type,
            version=version,
            model_params=model.get_params(),
            **(metadata or {}),
            **(additional_metadata or {}),
        )

        # Create version directory
        version_dir = self.models_dir / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension
        if self.serializer.serialization_format == "joblib":
            ext = ".joblib"
        else:
            ext = ".pkl"

        model_filename = f"{model_name}_v{version}{ext}"
        model_path = version_dir / model_filename

        # Serialize model
        self.serializer.serialize(model, model_path, compress=self.compress)

        # Save metadata
        metadata_path = version_dir / f"{model_name}_v{version}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(model_metadata.to_dict(), f, indent=2)

        # Register in registry
        self.registry.register(model_name, version, model_path, model_metadata)

        logger.info(f"Model {model_name} version {version} saved successfully")
        return model_path

    def load(
        self,
        model_name: str,
        version: Optional[str] = None,
        return_metadata: bool = False,
    ) -> Union[BaseEstimator, Tuple[BaseEstimator, ModelMetadata]]:
        """Load model by name and version.

        Args:
            model_name: Name of the model
            version: Version string (default: latest)
            return_metadata: Whether to return metadata (default: False)

        Returns:
            Loaded model, or tuple of (model, metadata) if return_metadata=True

        Raises:
            FileNotFoundError: If model not found
            ValueError: If loading fails
        """
        if version is None:
            version = self.registry.get_latest_version(model_name)
            if version is None:
                raise FileNotFoundError(f"No versions found for model: {model_name}")

        model_info = self.registry.get_model_info(model_name, version)
        if model_info is None:
            raise FileNotFoundError(
                f"Model {model_name} version {version} not found in registry"
            )

        model_path = Path(model_info["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = self.serializer.deserialize(model_path)

        if return_metadata:
            metadata_path = Path(model_info["metadata_path"])
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata_dict = json.load(f)
                metadata = ModelMetadata.from_dict(metadata_dict)
            else:
                metadata = ModelMetadata.from_dict(model_info["metadata"])

            return model, metadata

        return model

    def get_metadata(
        self, model_name: str, version: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """Get model metadata.

        Args:
            model_name: Name of the model
            version: Version string (default: latest)

        Returns:
            ModelMetadata instance or None
        """
        if version is None:
            version = self.registry.get_latest_version(model_name)
            if version is None:
                return None

        model_info = self.registry.get_model_info(model_name, version)
        if model_info is None:
            return None

        metadata_path = Path(model_info["metadata_path"])
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
            return ModelMetadata.from_dict(metadata_dict)
        else:
            return ModelMetadata.from_dict(model_info["metadata"])

    def list_models(self) -> List[str]:
        """List all registered models.

        Returns:
            List of model names
        """
        return self.registry.list_models()

    def list_versions(self, model_name: str) -> List[str]:
        """List all versions for a model.

        Args:
            model_name: Name of the model

        Returns:
            List of version strings
        """
        return self.registry.get_versions(model_name)

    def compare_versions(
        self, model_name: str, versions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare metadata across model versions.

        Args:
            model_name: Name of the model
            versions: List of versions to compare (default: all versions)

        Returns:
            DataFrame with version comparison
        """
        if versions is None:
            versions = self.list_versions(model_name)

        comparison_data = []
        for version in versions:
            metadata = self.get_metadata(model_name, version)
            if metadata:
                comparison_data.append(metadata.to_dict())

        if not comparison_data:
            return pd.DataFrame()

        return pd.DataFrame(comparison_data)

    def delete_version(self, model_name: str, version: str) -> None:
        """Delete a specific model version.

        Args:
            model_name: Name of the model
            version: Version string to delete

        Raises:
            FileNotFoundError: If version not found
        """
        model_info = self.registry.get_model_info(model_name, version)
        if model_info is None:
            raise FileNotFoundError(
                f"Model {model_name} version {version} not found"
            )

        # Delete model file
        model_path = Path(model_info["model_path"])
        if model_path.exists():
            model_path.unlink()

        # Delete metadata file
        metadata_path = Path(model_info["metadata_path"])
        if metadata_path.exists():
            metadata_path.unlink()

        # Remove from registry
        registry = self.registry._load_registry()
        if model_name in registry and version in registry[model_name]:
            del registry[model_name][version]
            if not registry[model_name]:
                del registry[model_name]
            self.registry._save_registry(registry)

        logger.info(f"Deleted {model_name} version {version}")


class ModelPersistenceManager:
    """High-level model persistence manager with configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize model persistence manager.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.persistence = None
        self._initialize_persistence()

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
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

    def _initialize_persistence(self) -> None:
        """Initialize persistence based on config."""
        persistence_config = self.config.get("persistence", {})
        models_dir = persistence_config.get("models_dir", "models")
        registry_path = persistence_config.get("registry_path", None)
        serialization_format = persistence_config.get("serialization_format", "joblib")
        compress = persistence_config.get("compress", True)

        self.persistence = ModelPersistence(
            models_dir=models_dir,
            registry_path=registry_path,
            serialization_format=serialization_format,
            compress=compress,
        )

    def save_model(
        self,
        model: BaseEstimator,
        model_name: str,
        version: str,
        **metadata,
    ) -> Path:
        """Save model with versioning.

        Args:
            model: Model to save
            model_name: Name of the model
            version: Version string
            **metadata: Additional metadata fields

        Returns:
            Path to saved model file
        """
        return self.persistence.save(
            model, model_name, version, additional_metadata=metadata
        )

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        return_metadata: bool = False,
    ) -> Union[BaseEstimator, Tuple[BaseEstimator, ModelMetadata]]:
        """Load model by name and version.

        Args:
            model_name: Name of the model
            version: Version string (default: latest)
            return_metadata: Whether to return metadata (default: False)

        Returns:
            Loaded model, or tuple of (model, metadata) if return_metadata=True
        """
        return self.persistence.load(
            model_name, version, return_metadata=return_metadata
        )

    def get_model_metadata(
        self, model_name: str, version: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """Get model metadata.

        Args:
            model_name: Name of the model
            version: Version string (default: latest)

        Returns:
            ModelMetadata instance or None
        """
        return self.persistence.get_metadata(model_name, version)


def main():
    """Main entry point for model persistence CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Model persistence and versioning with metadata tracking"
    )
    parser.add_argument(
        "action",
        choices=["save", "load", "list", "info", "compare", "delete"],
        help="Action to perform",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name of the model",
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Model version",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        help="Path to model file (for save action)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (for load action)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to metadata JSON file (for save action)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    manager = ModelPersistenceManager(
        config_path=Path(args.config) if args.config else None
    )

    if args.action == "save":
        if not args.model_name or not args.version or not args.model_file:
            print("Error: --model-name, --version, and --model-file are required")
            return

        import pickle

        with open(args.model_file, "rb") as f:
            model = pickle.load(f)

        metadata = {}
        if args.metadata:
            with open(args.metadata, "r") as f:
                metadata = json.load(f)

        model_path = manager.save_model(model, args.model_name, args.version, **metadata)
        print(f"Model saved to: {model_path}")

    elif args.action == "load":
        if not args.model_name:
            print("Error: --model-name is required")
            return

        model = manager.load_model(args.model_name, args.version)
        print(f"Model {args.model_name} version {args.version or 'latest'} loaded")

        if args.output:
            import pickle

            with open(args.output, "wb") as f:
                pickle.dump(model, f)
            print(f"Model saved to: {args.output}")

    elif args.action == "list":
        models = manager.persistence.list_models()
        if models:
            print("Registered models:")
            for model_name in models:
                versions = manager.persistence.list_versions(model_name)
                print(f"  {model_name}: {len(versions)} version(s)")
                for version in versions:
                    print(f"    - {version}")
        else:
            print("No models registered")

    elif args.action == "info":
        if not args.model_name:
            print("Error: --model-name is required")
            return

        metadata = manager.get_model_metadata(args.model_name, args.version)
        if metadata:
            print(f"\nModel: {metadata.model_name}")
            print(f"Type: {metadata.model_type}")
            print(f"Version: {metadata.version}")
            print(f"Created: {metadata.created_at}")
            print(f"Updated: {metadata.updated_at}")
            print("\nAdditional metadata:")
            for key, value in metadata.metadata.items():
                print(f"  {key}: {value}")
        else:
            print(f"Model {args.model_name} not found")

    elif args.action == "compare":
        if not args.model_name:
            print("Error: --model-name is required")
            return

        comparison = manager.persistence.compare_versions(args.model_name)
        if not comparison.empty:
            print(f"\nVersion comparison for {args.model_name}:")
            print(comparison.to_string(index=False))
        else:
            print(f"No versions found for {args.model_name}")

    elif args.action == "delete":
        if not args.model_name or not args.version:
            print("Error: --model-name and --version are required")
            return

        manager.persistence.delete_version(args.model_name, args.version)
        print(f"Deleted {args.model_name} version {args.version}")


if __name__ == "__main__":
    main()
