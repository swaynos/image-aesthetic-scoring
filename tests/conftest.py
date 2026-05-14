"""Pytest configuration: fixtures and path constants."""
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

# ── v1 aesthetic scoring fixtures ─────────────────────────────────────────────
PHOTO_A = str(FIXTURES / "photo_a.jpg")
PHOTO_B = str(FIXTURES / "photo_b.jpg")
PROMPT  = (FIXTURES / "prompt.txt").read_text().strip()
