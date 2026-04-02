"""Vision integration test — exercises multimodal image understanding via Ollama.

Run with:  pytest tests/test_vision.py -v -s
Requires:  A running Ollama server with a vision model (llama3.2-vision).

Each test sends an image to the model and asserts the response contains
the expected description.
"""

import os
from pathlib import Path

import pytest

from pygentix import Ollama

TESTS_DIR = Path(__file__).parent
PHOTO1 = str(TESTS_DIR / "photo1.jpeg")
PHOTO2 = str(TESTS_DIR / "photo2.jpeg")
PHOTO3 = str(TESTS_DIR / "photo3.jpeg")

VISION_MODEL = "llama3.2-vision"


def _is_vision_available() -> bool:
    try:
        from ollama import list as ollama_list

        response = ollama_list()
        models = {
            getattr(m, "model", None) or getattr(m, "name", None)
            for m in (getattr(response, "models", []) or [])
        }
        return any(m.startswith(VISION_MODEL) for m in models if m)
    except Exception:
        return False


requires_vision = pytest.mark.skipif(
    not _is_vision_available(),
    reason=f"Ollama vision model '{VISION_MODEL}' not available",
)


@pytest.fixture
def agent():
    a = Ollama(model=VISION_MODEL)
    return a


@requires_vision
class TestVision:

    def test_photo1_cats(self, agent):
        """photo1.jpeg shows two cats — the model should mention 'cat'."""
        conv = agent.start_conversation()
        response = conv.ask("What do you see in this image?", images=[PHOTO1])
        answer = response.message.content.lower()
        assert "cat" in answer, f"Expected 'cat' in answer: {answer}"

    def test_photo2_three_cats(self, agent):
        """photo2.jpeg shows exactly three cats."""
        conv = agent.start_conversation()
        response = conv.ask(
            "How many cats are in this photo? Reply with just the number.",
            images=[PHOTO2],
        )
        answer = response.message.content.lower()
        assert "3" in answer or "three" in answer, (
            f"Expected '3' or 'three' in answer: {answer}"
        )

    def test_photo3_rainbow(self, agent):
        """photo3.jpeg shows a rainbow — the model should mention it."""
        conv = agent.start_conversation()
        response = conv.ask("What do you see in this image?", images=[PHOTO3])
        answer = response.message.content.lower()
        assert "rainbow" in answer, f"Expected 'rainbow' in answer: {answer}"
