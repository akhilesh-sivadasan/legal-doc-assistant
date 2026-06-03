"""
pdf_scraper.py
==============

Stage 1 of the pipeline: turn the raw Criminal Code PDF into structured,
machine-readable section records.

The Canadian Criminal Code is published as a single large PDF (Justice Laws
file ``C-46.pdf``). For a search/RAG system we don't want one giant blob of
text; we want individually addressable "sections" (e.g. section 380 "Fraud")
so that search results can cite a specific provision and page.

This module walks the PDF page by page, uses a regex heuristic to detect where
each numbered section begins, and accumulates the lines that follow until the
next section header appears. The result is written to
``data/processed/sections.json`` for the indexing stage (``build_index.py``)
to consume.

Known limitation — bilingual layout
------------------------------------
The official PDF is bilingual: English and French provisions are interleaved
(the French text usually follows the English on the same or adjacent pages).
Because we extract text linearly with PyMuPDF, a single "section" record can
contain both the English and the French wording. This is acceptable for a
semantic search prototype (the embedding model still matches the English
query against the English portion), but it is why ``content`` fields in the
output often look like they contain duplicated/translated text. Splitting the
two languages cleanly would require column-aware layout parsing and is out of
scope here.

Pipeline position: ``pdf_scraper.py`` -> ``build_index.py`` -> ``app.py``.
"""

import fitz  # PyMuPDF — used purely for fast, layout-agnostic text extraction.
import re
import json
from pathlib import Path
from typing import List, Dict
import logging

# A single shared logger for the whole module. We log progress (page counts,
# extracted totals) at INFO so a user running the script from the CLI gets
# feedback on a long-running parse without needing a progress bar.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class CriminalCodeScraper:
    """Extracts numbered sections from the Criminal Code PDF.

    The class is deliberately stateless between runs: you point it at a PDF
    path, call :meth:`run`, and it writes ``sections.json``. All tuning knobs
    (validation thresholds, the section-header regex) live in the methods that
    use them so the parsing strategy stays readable in one place.
    """

    def __init__(self, pdf_path: str):
        # Resolve everything to Path objects up front so the rest of the class
        # never has to worry about str-vs-Path or platform path separators.
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path("data/processed")
        self.output_file = self.output_dir / "sections.json"

        # Create the output directory eagerly. Doing it here (rather than at
        # save time) means a misconfigured output location fails fast, before
        # we spend time parsing a multi-hundred-page PDF.
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_pdf(self) -> bool:
        """Return True only if the source PDF actually exists on disk.

        We check existence explicitly (instead of letting ``fitz.open`` raise)
        so we can emit a clear, actionable log message — the most common setup
        mistake is forgetting to download C-46.pdf into ``data/``.
        """
        if not self.pdf_path.exists():
            logger.error(f"PDF not found: {self.pdf_path}")
            return False
        return True

    def extract_sections(self) -> List[Dict[str, str]]:
        """Parse the PDF into a list of raw section dicts.

        Strategy: stream the document one page at a time, one line at a time.
        We keep a "current section" plus a buffer of the lines seen since its
        header. When we hit a line that looks like a *new* section header, we
        finalise the previous section (join its buffered lines into ``content``)
        and start fresh. The final section is flushed after the loop ends.

        Returns an empty list on any failure so callers can simply check for
        truthiness rather than handling exceptions.
        """
        if not self.validate_pdf():
            return []

        sections: List[Dict[str, str]] = []
        current_section = None      # The header dict we're currently filling in.
        section_buffer: List[str] = []  # Lines accumulated for current_section.

        try:
            doc = fitz.open(str(self.pdf_path))
            logger.info(f"Processing {len(doc)} pages...")

            for page_num, page in enumerate(doc):
                # PyMuPDF returns the page's text in roughly reading order.
                # Splitting on newlines gives us line-level granularity, which
                # is the unit our header heuristic operates on.
                text = page.get_text()
                lines = text.split('\n')

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue  # Skip blank lines — they carry no signal.

                    # Header heuristic: a section starts with a section number
                    # followed by a title. The number can be:
                    #   - 1-3 digits ("85", "380")
                    #   - optionally one trailing capital letter ("85A")
                    #   - optionally a dotted sub-number ("487.01")
                    # Group 1 captures the id; group 2 captures the title text.
                    section_match = re.match(r'^(\d{1,3}[A-Z]?(?:\.\d+)?)\s+(.+)', line)

                    if section_match and self._is_valid_section_start(line):
                        # We've found a new header, so the previous section is
                        # complete. Join its buffered lines into one content
                        # string and keep it only if it passes quality checks.
                        if current_section and section_buffer:
                            current_section["content"] = '\n'.join(section_buffer)
                            if self._is_valid_section(current_section):
                                sections.append(current_section)

                        # Begin accumulating the new section. We record the
                        # 1-based page number so the UI can show "p. 42".
                        current_section = {
                            "id": section_match.group(1),
                            "title": section_match.group(2).strip(),
                            "page_start": page_num + 1
                        }
                        section_buffer = [line]

                    elif current_section:
                        # Not a header — it's body text belonging to the section
                        # we're currently inside. (Lines before the very first
                        # header, e.g. cover pages, are intentionally dropped.)
                        section_buffer.append(line)

            # Flush the last section, which has no following header to trigger
            # the save inside the loop.
            if current_section and section_buffer:
                current_section["content"] = '\n'.join(section_buffer)
                if self._is_valid_section(current_section):
                    sections.append(current_section)

            doc.close()
            logger.info(f"Extracted {len(sections)} sections")
            return sections

        except Exception as e:
            # Any PyMuPDF/IO error here is non-recoverable for this run; log it
            # and return empty so the CLI exits with a clear failure.
            logger.error(f"Extraction failed: {e}")
            return []

    def _is_valid_section_start(self, line: str) -> bool:
        """Reject lines that match the header regex but aren't real headers.

        The header regex (`number + text`) is intentionally loose, so it also
        matches things like page footers ("12 Page 3") or subsection callouts
        ("5 (2) ..."). This method is the second filter that removes those
        false positives. Returning False means "treat this as body text".
        """
        # A genuine "<number> <title>" header is never this short; tiny lines
        # are almost always stray numbers, margin notes, or artefacts.
        if len(line.strip()) < 10:
            return False

        # Patterns that look like headers to the regex but are not. Kept as a
        # list so new false-positive shapes can be added without touching logic.
        false_positives = [
            r'^\d+\s*$',           # A bare number on its own line.
            r'^\d+\s+Page\s+\d+',  # Page-reference footers ("12 Page 3").
            r'^\d+\s+\(\d+\)',     # Subsection references ("5 (2)").
        ]

        for pattern in false_positives:
            if re.match(pattern, line, re.IGNORECASE):
                return False
        return True

    def _is_valid_section(self, section: Dict[str, str]) -> bool:
        """Drop sections that are too thin to be useful for search.

        Thresholds are deliberately conservative: a real provision has at least
        a short title and a sentence or two of body text. Headers with almost
        no content are usually parsing noise (e.g. a marginal note that got
        mistaken for a section start) and would only pollute the search index.
        """
        content = section.get("content", "").strip()
        title = section.get("title", "").strip()
        return len(content) >= 50 and len(title) >= 5

    def clean_sections(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Normalise whitespace and annotate each section with a word count.

        Embeddings and display both benefit from single-spaced text, so we
        collapse every run of whitespace (newlines included) into one space.
        ``word_count`` is stored for lightweight downstream stats/filtering.
        Mutates and returns the same dicts — there's no need for copies because
        these objects exist only within a single pipeline run.
        """
        cleaned = []
        for section in sections:
            # ``\s+`` already collapses newlines, tabs and repeated spaces in
            # one pass, so we feed it the raw content directly.
            content = re.sub(r'\s+', ' ', section["content"]).strip()

            section["content"] = content
            section["word_count"] = len(content.split())
            cleaned.append(section)
        return cleaned

    def save_sections(self, sections: List[Dict[str, str]]) -> bool:
        """Write the sections to ``sections.json`` with a small metadata header.

        Returns False (without writing) when there's nothing to save, so an
        empty/failed extraction never overwrites a previously good file.
        """
        if not sections:
            return False

        try:
            # Wrap the list in an object with a ``metadata`` block. build_index
            # accepts both a bare list and this dict shape, but the metadata is
            # handy for provenance/debugging ("which PDF produced this?").
            output_data = {
                "metadata": {
                    "source_file": str(self.pdf_path),
                    "total_sections": len(sections),
                },
                "sections": sections
            }

            # ensure_ascii=False keeps accented French characters readable in
            # the JSON; indent=2 keeps the file diff-friendly under version
            # control despite its size.
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved to {self.output_file}")
            return True

        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False

    def run(self) -> bool:
        """Run the full extract -> clean -> save pipeline.

        Returns True on success, False if any stage produced nothing. Designed
        so the CLI entry point can map the boolean straight onto an exit code.
        """
        sections = self.extract_sections()
        if not sections:
            return False

        sections = self.clean_sections(sections)
        return self.save_sections(sections)


if __name__ == "__main__":
    # CLI entry point: parse the default Criminal Code file and report status
    # via the process exit code (0 = success, 1 = failure) so the step can be
    # chained in a shell script or CI before build_index.py runs.
    scraper = CriminalCodeScraper("data/C-46.pdf")
    if scraper.run():
        print("Done.")
    else:
        print("Failed.")
        exit(1)
