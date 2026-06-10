"""
build_index.py
==============

Stage 2 of the pipeline: turn the structured sections produced by
``pdf_scraper.py`` into a FAISS vector index that the app can search.

What "indexing" means here
--------------------------
Semantic search works by converting each piece of text into a high-dimensional
vector (an "embedding") such that texts with similar meaning end up close
together in that space. At query time we embed the user's question the same way
and ask FAISS for the nearest section vectors. This module performs the
one-time, offline work of:

  1. Loading the cleaned sections from ``data/processed/sections.json``.
  2. Wrapping each one in a LangChain ``Document`` (text + metadata).
  3. Embedding them with a local sentence-transformer model.
  4. Saving the resulting FAISS index to ``faiss_index/`` for ``app.py``.

The embedding model (``all-MiniLM-L6-v2``) runs locally on CPU, so building the
index requires no API key and no network beyond the initial model download.

Pipeline position: ``pdf_scraper.py`` -> ``build_index.py`` -> ``app.py``.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# load_dotenv is called for consistency with the rest of the project (so any
# future env-based config Just Works), even though indexing currently needs no
# secrets — the embedding model is local.
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class IndexBuilder:
    """Builds and persists the FAISS index from extracted sections."""

    def __init__(self):
        # Input/output locations and the embedding model are fixed by
        # convention. They're attributes (not constants) so a test or a
        # notebook could instantiate the builder and override them if needed.
        self.sections_file = Path("data/processed/sections.json")
        self.index_dir = Path("faiss_index")

        # all-MiniLM-L6-v2 is a small (~80MB), fast, well-rounded sentence
        # embedding model. It's a good default for prototypes: low memory, no
        # GPU required, and "good enough" semantic quality for legal lookup.
        # IMPORTANT: this MUST match the model used in app.py at query time —
        # an index built with one model is meaningless to another.
        self.embedding_model = "all-MiniLM-L6-v2"

    def load_sections(self) -> Optional[List[Dict]]:
        """Load sections from JSON, tolerating both supported file shapes.

        ``pdf_scraper.py`` writes ``{"metadata": {...}, "sections": [...]}``,
        but we also accept a bare top-level list so the index can be rebuilt
        from hand-edited or alternative section files. Returns None on any
        problem so the caller can abort cleanly.
        """
        if not self.sections_file.exists():
            logger.error("Sections file not found. Run pdf_scraper.py first.")
            return None

        try:
            with open(self.sections_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Accept either the wrapped dict form or a raw list.
            sections = data.get("sections") if isinstance(data, dict) else data
            logger.info(f"Loaded {len(sections)} sections")
            return sections

        except Exception as e:
            logger.error(f"Error loading sections: {e}")
            return None

    def create_documents(self, sections: List[Dict]) -> List[Document]:
        """Convert raw section dicts into LangChain ``Document`` objects.

        Each Document carries the searchable ``page_content`` plus a ``metadata``
        dict. The metadata is preserved verbatim through FAISS, so anything we
        attach here (id, title, page, category) is available on search results
        in the UI without a second lookup.

        Sections missing the essentials (``content`` or ``id``) are skipped —
        an entry with no id can't be cited, and one with no content can't be
        meaningfully embedded.
        """
        documents = []
        for s in sections:
            if not s.get("content") or not s.get("id"):
                continue

            doc = Document(
                page_content=s["content"],
                metadata={
                    "id": s["id"],
                    "title": s.get("title", ""),
                    "page_start": s.get("page_start", 0),
                    # Derive a coarse topical category from the section number.
                    # This is surfaced in the app's "Sources" tabs so users can
                    # see at a glance which part of the Code a hit comes from.
                    "section_type": self._get_category(s["id"])
                }
            )
            documents.append(doc)

        return documents

    def _get_category(self, section_id: str) -> str:
        """Map a section id to a broad subject area of the Criminal Code.

        The Code is organised so that section-number ranges correspond to
        themes (offences against the person, property offences, procedure,
        etc.). We approximate that grouping from the leading integer of the id.
        These boundaries are deliberately coarse — they're for UI labelling and
        rough filtering, not legal precision.

        ``section_id`` can look like "85", "85A" or "487.01"; we take the part
        before the first dot and strip any trailing letter to get the integer.
        """
        try:
            num = int(section_id.split('.')[0].rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
            if num <= 42:  return "general"
            if num <= 130: return "public_order"
            if num <= 200: return "person_reputation"
            if num <= 370: return "property"
            if num <= 490: return "fraud"
            if num <= 672: return "procedure"
            return "other"
        except (ValueError, AttributeError):
            # ValueError: the id had no parseable leading integer.
            # AttributeError: section_id wasn't a string.
            # Either way the id is malformed, so bucket it as "other" rather
            # than letting a bad record crash the whole build. (We catch
            # specific exceptions, not a bare ``except``, so genuine bugs and
            # interrupts like Ctrl-C still propagate.)
            return "other"

    def build_index(self, documents: List[Document]) -> bool:
        """Embed all documents and persist the FAISS index to disk.

        Returns True on success, False if there was nothing to index or the
        embedding/save failed.
        """
        try:
            logger.info(f"Embedding {len(documents)} documents...")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                # Normalising embeddings to unit length means cosine similarity
                # reduces to a dot product, which is what FAISS uses by default
                # — this keeps query-time ranking consistent and well-behaved.
                encode_kwargs={'normalize_embeddings': True}
            )

            # Embed in batches rather than all at once. For a few thousand short
            # sections this isn't strictly necessary, but it caps peak memory
            # and gives us periodic progress logs on slower machines. We build a
            # FAISS index from the first batch, then merge each subsequent batch
            # into it.
            batch_size = 100
            db = None

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                if db is None:
                    db = FAISS.from_documents(batch, embeddings)
                else:
                    db.merge_from(FAISS.from_documents(batch, embeddings))

                # Log every 5th batch (every ~500 docs) to avoid log spam while
                # still showing forward progress on a long build.
                if (i // batch_size) % 5 == 0:
                    logger.info(f"Processed {min(i + batch_size, len(documents))}/{len(documents)}")

            if db:
                # save_local writes two files: index.faiss (the vectors) and
                # index.pkl (the docstore + id mapping). app.py loads both.
                db.save_local(str(self.index_dir))
                logger.info(f"Index saved to {self.index_dir}")
                return True
            return False

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return False

    def run(self):
        """Run the full load -> document -> embed -> save pipeline."""
        sections = self.load_sections()
        if not sections:
            return  # load_sections already logged the reason.

        docs = self.create_documents(sections)
        self.build_index(docs)


if __name__ == "__main__":
    IndexBuilder().run()
