
import unittest
from pathlib import Path
import json

class TestDataIntegrity(unittest.TestCase):
    def test_pdf_exists(self):
        self.assertTrue(Path("data/C-46.pdf").exists(), "PDF file not found")

    def test_sections_json_exists(self):
        self.assertTrue(Path("data/processed/sections.json").exists(), "sections.json not found")

    def test_sections_json_content(self):
        with open("data/processed/sections.json", "r") as f:
            data = json.load(f)
            self.assertIn("sections", data)
            self.assertTrue(len(data["sections"]) > 0)

    def test_faiss_index_exists(self):
        self.assertTrue(Path("faiss_index/index.faiss").exists(), "index.faiss not found")
        self.assertTrue(Path("faiss_index/index.pkl").exists(), "index.pkl not found")

if __name__ == '__main__':
    unittest.main()
