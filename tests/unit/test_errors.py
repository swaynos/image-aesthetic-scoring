"""Unit tests for errors module."""
import pytest
from aesthetic_scoring.errors import ModelLoadError, ModelInferenceError, GpuMemoryError


def test_model_load_error_is_exception():
    with pytest.raises(ModelLoadError):
        raise ModelLoadError("test")


def test_model_inference_error_is_exception():
    with pytest.raises(ModelInferenceError):
        raise ModelInferenceError("test")


def test_gpu_memory_error_is_exception():
    with pytest.raises(GpuMemoryError):
        raise GpuMemoryError("test")


def test_all_inherit_from_exception():
    assert issubclass(ModelLoadError, Exception)
    assert issubclass(ModelInferenceError, Exception)
    assert issubclass(GpuMemoryError, Exception)
