"""
Builds FAISS index from extracted sections.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self):
        self.sections_file = Path("data/processed/sections.json")
        self.index_dir = Path("faiss_index")
        self.embedding_model = "all-MiniLM-L6-v2"
        
    def load_sections(self) -> Optional[List[Dict]]:
        if not self.sections_file.exists():
            logger.error("Sections file not found. Run pdf_scraper.py first.")
            return None
        
        try:
            with open(self.sections_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle list or dict format
            sections = data.get("sections") if isinstance(data, dict) else data
            logger.info(f"Loaded {len(sections)} sections")
            return sections
            
        except Exception as e:
            logger.error(f"Error loading sections: {e}")
            return None
    
    def create_documents(self, sections: List[Dict]) -> List[Document]:
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
                    "section_type": self._get_category(s["id"])
                }
            )
            documents.append(doc)
        
        return documents
    
    def _get_category(self, section_id: str) -> str:
        """Categorize based on section number range."""
        try:
            num = int(section_id.split('.')[0].rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
            if num <= 42: return "general"
            if num <= 130: return "public_order"
            if num <= 200: return "person_reputation"
            if num <= 370: return "property"
            if num <= 490: return "fraud"
            if num <= 672: return "procedure"
            return "other"
        except:
            return "other"
    
    def build_index(self, documents: List[Document]) -> bool:
        try:
            logger.info(f"Embedding {len(documents)} documents...")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Batch processing for memory efficiency
            batch_size = 100
            db = None
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                if db is None:
                    db = FAISS.from_documents(batch, embeddings)
                else:
                    db.merge_from(FAISS.from_documents(batch, embeddings))

                if (i // batch_size) % 5 == 0:
                    logger.info(f"Processed {min(i+batch_size, len(documents))}/{len(documents)}")
            
            if db:
                db.save_local(str(self.index_dir))
                logger.info(f"Index saved to {self.index_dir}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return False

    def run(self):
        sections = self.load_sections()
        if not sections: return
        
        docs = self.create_documents(sections)
        self.build_index(docs)

if __name__ == "__main__":
    IndexBuilder().run()
