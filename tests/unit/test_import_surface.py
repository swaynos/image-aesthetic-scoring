"""Test the public import surface of the package."""


def test_package_imports():
    import aesthetic_scoring  # noqa


def test_scoring_functions_exported():
    from aesthetic_scoring import (
        score_clipscore,
        score_fgaesq,
        score_hpsv2,
        score_laion,
        score_pickscore,
        score_images,
    )
    assert callable(score_laion)
    assert callable(score_pickscore)
    assert callable(score_hpsv2)
    assert callable(score_fgaesq)
    assert callable(score_clipscore)
    assert callable(score_images)


def test_result_types_exported():
    from aesthetic_scoring import (
        LaionScoreResult,
        PickScoreResult,
        HPSv2ScoreResult,
        CLIPScoreResult,
        FGAesQScoreResult,
        ImageScoreReport,
    )


def test_exceptions_exported():
    from aesthetic_scoring.errors import ModelInferenceError, GpuMemoryError, ModelLoadError


def test_types_importable_from_submodule():
    from aesthetic_scoring.types import (
        BaseScoreResult,
        LaionScoreResult,
        PickScoreResult,
        HPSv2ScoreResult,
        CLIPScoreResult,
        FGAesQScoreResult,
    )
