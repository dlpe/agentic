"""Vision integration tests — exercises multimodal image understanding via Ollama.

Run with:  pytest tests/test_vision.py -v -s
Requires:  A running Ollama server with a vision model (llama3.2-vision).

Tests cover photo recognition (cats, rainbow) and document parsing (PDF invoice
rendered to an image via PyMuPDF).
"""

from pathlib import Path

import pytest

from pygentix import Ollama

TESTS_DIR = Path(__file__).parent
PHOTO1 = str(TESTS_DIR / "photo1.jpeg")
PHOTO2 = str(TESTS_DIR / "photo2.jpeg")
PHOTO3 = str(TESTS_DIR / "photo3.jpeg")
INVOICE_PDF = str(TESTS_DIR / "invoice.pdf")

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
    return Ollama(model=VISION_MODEL)


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


# ---------------------------------------------------------------------------
# PDF document parsing
# ---------------------------------------------------------------------------

def _pdf_to_image(pdf_path: str) -> str:
    """Render the first page of *pdf_path* to a temporary PNG and return its path."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=200)
    out = pdf_path.replace(".pdf", "_rendered.png")
    pix.save(out)
    doc.close()
    return out


requires_pymupdf = pytest.mark.skipif(
    not _is_vision_available(),
    reason=f"Ollama vision model '{VISION_MODEL}' not available",
)

try:
    import fitz  # noqa: F401
    _has_pymupdf = True
except ImportError:
    _has_pymupdf = False

requires_pdf = pytest.mark.skipif(
    not (_is_vision_available() and _has_pymupdf),
    reason="Requires both vision model and PyMuPDF (pip install pymupdf)",
)


@requires_pdf
class TestPDFParsing:
    """Send a rendered PDF invoice to the vision model and verify extraction."""

    @pytest.fixture(autouse=True)
    def _setup(self, agent, tmp_path):
        self.agent = agent
        self.image = _pdf_to_image(INVOICE_PDF)

    def test_extract_company_name(self):
        """The model should identify the company name on the invoice."""
        conv = self.agent.start_conversation()
        response = conv.ask(
            "What is the company name at the top of this invoice? "
            "Reply with only the company name.",
            images=[self.image],
        )
        answer = response.message.content.lower()
        assert "techcorp" in answer, f"Expected 'techcorp' in answer: {answer}"

    def test_extract_total_amount(self):
        """The model should read the total amount from the invoice."""
        conv = self.agent.start_conversation()
        response = conv.ask(
            "What is the total amount on this invoice? "
            "Reply with only the dollar amount.",
            images=[self.image],
        )
        answer = response.message.content
        assert "5,454" in answer or "5454" in answer, (
            f"Expected '$5,454.00' in answer: {answer}"
        )

    def test_extract_invoice_number(self):
        """The model should read the invoice number."""
        conv = self.agent.start_conversation()
        response = conv.ask(
            "What is the invoice number on this document? "
            "Reply with only the invoice number.",
            images=[self.image],
        )
        answer = response.message.content.upper()
        assert "INV-2026-001" in answer, (
            f"Expected 'INV-2026-001' in answer: {answer}"
        )

    def test_count_line_items(self):
        """The invoice has 5 line items — vision models sometimes miscount by ±1."""
        conv = self.agent.start_conversation()
        response = conv.ask(
            "Count the product rows in the invoice table (exclude the header). "
            "Reply with only the number.",
            images=[self.image],
        )
        answer = response.message.content.strip().lower()
        acceptable = {"4", "5", "6", "four", "five", "six"}
        assert any(v in answer for v in acceptable), (
            f"Expected 4-6 line items, got: {answer}"
        )

    def test_extract_client_name(self):
        """The model should identify who the invoice is billed to."""
        conv = self.agent.start_conversation()
        response = conv.ask(
            "Who is this invoice billed to? Reply with only the client name.",
            images=[self.image],
        )
        answer = response.message.content.lower()
        assert "acme" in answer, f"Expected 'acme' in answer: {answer}"
