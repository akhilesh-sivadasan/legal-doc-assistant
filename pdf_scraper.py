"""
Extracts sections from the Canadian Criminal Code PDF.
"""

import fitz  # PyMuPDF
import re
import json
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class CriminalCodeScraper:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path("data/processed")
        self.output_file = self.output_dir / "sections.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_pdf(self) -> bool:
        if not self.pdf_path.exists():
            logger.error(f"PDF not found: {self.pdf_path}")
            return False
        return True
    
    def extract_sections(self) -> List[Dict[str, str]]:
        if not self.validate_pdf():
            return []
        
        sections = []
        current_section = None
        section_buffer = []
        
        try:
            doc = fitz.open(str(self.pdf_path))
            logger.info(f"Processing {len(doc)} pages...")
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Matches: "267", "267.1", "85A" followed by title
                    section_match = re.match(r'^(\d{1,3}[A-Z]?(?:\.\d+)?)\s+(.+)', line)
                    
                    if section_match and self._is_valid_section_start(line):
                        # Save previous section
                        if current_section and section_buffer:
                            current_section["content"] = '\n'.join(section_buffer)
                            if self._is_valid_section(current_section):
                                sections.append(current_section)
                        
                        # Start new section
                        current_section = {
                            "id": section_match.group(1),
                            "title": section_match.group(2).strip(),
                            "page_start": page_num + 1
                        }
                        section_buffer = [line]
                        
                    elif current_section:
                        section_buffer.append(line)
            
            # Save final section
            if current_section and section_buffer:
                current_section["content"] = '\n'.join(section_buffer)
                if self._is_valid_section(current_section):
                    sections.append(current_section)
            
            doc.close()
            logger.info(f"Extracted {len(sections)} sections")
            return sections
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return []
    
    def _is_valid_section_start(self, line: str) -> bool:
        """Validate if line is a section start (avoiding false positives)."""
        if len(line.strip()) < 10:
            return False
            
        false_positives = [
            r'^\d+\s*$',           # Just a number
            r'^\d+\s+Page\s+\d+',  # Page references
            r'^\d+\s+\(\d+\)',     # Subsection references
        ]
        
        for pattern in false_positives:
            if re.match(pattern, line, re.IGNORECASE):
                return False
        return True
    
    def _is_valid_section(self, section: Dict[str, str]) -> bool:
        """Filter out low-quality sections."""
        content = section.get("content", "").strip()
        title = section.get("title", "").strip()
        return len(content) >= 50 and len(title) >= 5
    
    def clean_sections(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        cleaned = []
        for section in sections:
            content = section["content"]
            # Normalize whitespace
            content = re.sub(r'\s+', ' ', content.replace('\n', ' ')).strip()
            
            section["content"] = content
            section["word_count"] = len(content.split())
            cleaned.append(section)
        return cleaned
    
    def save_sections(self, sections: List[Dict[str, str]]) -> bool:
        if not sections:
            return False
        
        try:
            output_data = {
                "metadata": {
                    "source_file": str(self.pdf_path),
                    "total_sections": len(sections),
                },
                "sections": sections
            }
            
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved to {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False
    
    def run(self) -> bool:
        sections = self.extract_sections()
        if not sections:
            return False
        
        sections = self.clean_sections(sections)
        return self.save_sections(sections)

if __name__ == "__main__":
    scraper = CriminalCodeScraper("data/C-46.pdf")
    if scraper.run():
        print("Done.")
    else:
        print("Failed.")
        exit(1)
