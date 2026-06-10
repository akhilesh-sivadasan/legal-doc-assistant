"""
tests/test_data.py
==================

Smoke tests for the data artefacts the application depends on.

These are intentionally lightweight *integrity* checks, not unit tests of the
parsing/embedding logic. Their job is to catch the most common "it doesn't run"
failures before the app is launched:

  - the source PDF is missing,
  - the scraper hasn't been run (no sections.json),
  - sections.json is present but empty/malformed,
  - the FAISS index hasn't been built.

Deeper unit tests of the pure logic (e.g. the section-header regex or the
category bucketing) are deliberately out of scope here, because importing the
modules that contain them pulls in heavy optional dependencies (PyMuPDF,
LangChain, FAISS) that may not be installed in a minimal environment. Keeping
these tests dependency-free means they always run.

Run with:  python -m unittest tests/test_data.py
"""

import unittest
from pathlib import Path
import json


class TestDataIntegrity(unittest.TestCase):
    """Verify the pipeline's on-disk artefacts exist and are well-formed."""

    def test_pdf_exists(self):
        # The raw Criminal Code PDF is the input to the whole pipeline; without
        # it pdf_scraper.py has nothing to parse.
        self.assertTrue(Path("data/C-46.pdf").exists(), "PDF file not found")

    def test_sections_json_exists(self):
        # Produced by pdf_scraper.py and consumed by build_index.py — its
        # absence means stage 1 hasn't been run.
        self.assertTrue(Path("data/processed/sections.json").exists(), "sections.json not found")

    def test_sections_json_content(self):
        # Beyond existence, confirm the file parses as JSON, uses the expected
        # ``sections`` key, and actually contains at least one section. An empty
        # or differently-shaped file would silently produce an empty index.
        with open("data/processed/sections.json", "r") as f:
            data = json.load(f)
            self.assertIn("sections", data)
            self.assertTrue(len(data["sections"]) > 0)

    def test_faiss_index_exists(self):
        # save_local() writes exactly these two files: the vectors (.faiss) and
        # the docstore/id map (.pkl). app.py needs both to load the index.
        self.assertTrue(Path("faiss_index/index.faiss").exists(), "index.faiss not found")
        self.assertTrue(Path("faiss_index/index.pkl").exists(), "index.pkl not found")


if __name__ == '__main__':
    unittest.main()
