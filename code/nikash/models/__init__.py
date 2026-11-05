"""Model backends: MockModel (CPU, smoke test) and HFModel (Kaggle GPU, imported lazily)."""

from .mock_backend import MockModel, default_mock_models

__all__ = ["MockModel", "default_mock_models"]
