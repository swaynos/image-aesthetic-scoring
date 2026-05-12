"""Custom exceptions for the aesthetic_scoring library."""


class ModelLoadError(Exception):
    """Raised when a model fails to load or initialize."""


class ModelInferenceError(Exception):
    """Raised when inference fails for a reason other than CUDA OOM."""


class GpuMemoryError(Exception):
    """Raised when CUDA runs out of memory during inference or loading."""
