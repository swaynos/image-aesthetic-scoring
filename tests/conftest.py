"""Pytest configuration: fixtures and markers."""
import os
import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
PHOTO_A = os.path.join(FIXTURE_DIR, "photo_a.jpg")
PHOTO_B = os.path.join(FIXTURE_DIR, "photo_b.jpg")
PROMPT = "a vibrant colorful gradient abstract image"


@pytest.fixture
def photo_a():
    return PHOTO_A


@pytest.fixture
def photo_b():
    return PHOTO_B


@pytest.fixture
def fixture_prompt():
    return PROMPT


@pytest.fixture
def two_photos():
    return [PHOTO_A, PHOTO_B]
