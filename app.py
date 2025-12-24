"""
Canadian Criminal Code Search
Streamlit application for searching and summarizing Canadian Criminal Code sections.
"""

import streamlit as st
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Canadian Criminal Code Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LegalSearchApp:
    def __init__(self):
        self._setup_api()
        
    def _setup_api(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY not found in environment variables")
            st.stop()
        genai.configure(api_key=api_key)

    @st.cache_resource
    def load_models(_self):
        """Load embeddings and FAISS index."""
        try:
            with st.spinner("Loading search index..."):
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                index_path = Path("faiss_index")
                if not index_path.exists():
                    st.error("Search index not found. Run 'python build_index.py' first.")
                    st.stop()
                
                db = FAISS.load_local(
                    str(index_path), 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                return db
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            st.error("System initialization failed. Check logs.")
            st.stop()

    @st.cache_data
    def search_documents(_self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents from FAISS."""
        try:
            db = _self.load_models()
            retriever = db.as_retriever(search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(query)
            
            return [{
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in docs]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def generate_answer(self, query: str, context: str) -> Optional[str]:
        """Generate response using Gemini."""
        prompt = f"""You are a legal assistant specializing in the Canadian Criminal Code.
Use the provided legal text to answer the question accurately.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Cite specific sections where applicable.
- State clearly if the context is insufficient.
- Maintain a professional, neutral tone.
- Disclaimer: This is for information only, not legal advice.

ANSWER:"""

        try:
            model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-001")
            response = model.generate_content([prompt])
            return response.text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    def process_upload(self, uploaded_file) -> Optional[str]:
        try:
            content = uploaded_file.read().decode("utf-8")
            if len(content.strip()) < 100:
                st.warning("File content too short for analysis.")
            return content
        except Exception as e:
            logger.error(f"Upload processing failed: {e}")
            st.error("Failed to read file.")
            return None

def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []

def render_sidebar(app: LegalSearchApp) -> Optional[str]:
    with st.sidebar:
        st.header("Settings")
        
        uploaded_file = st.file_uploader(
            "Context Analysis",
            type=["txt"],
            help="Upload a specific legal document to analyze."
        )
        
        context_text = None
        if uploaded_file:
            context_text = app.process_upload(uploaded_file)
            if context_text:
                st.success(f"Loaded {len(context_text)} chars")
        
        st.divider()
        st.subheader("Recent Queries")
        
        if st.session_state.history:
            for q, _, ts in st.session_state.history[-5:]:
                st.caption(f"{ts} - {q[:30]}...")

            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()
        else:
            st.caption("No history.")

        return context_text

def render_main(app: LegalSearchApp, context_text: Optional[str]):
    st.title("Canadian Criminal Code Search")
    st.caption("Search the Criminal Code and get AI-summarized answers with citations.")
    st.divider()
    
    query = st.text_input("Search Query", placeholder="e.g., What is the penalty for fraud over $5000?")
    
    if query and query.strip():
        with st.spinner("Analyzing..."):
            if context_text:
                context = context_text
                sources = []
                st.info("Using uploaded context.")
            else:
                sources = app.search_documents(query)
                context = "\n\n".join([d["content"] for d in sources])
            
            if not context.strip():
                st.warning("No relevant information found.")
                return

            answer = app.generate_answer(query, context)

            if answer:
                st.subheader("Summary")
                st.info(answer)
                
                if sources:
                    st.subheader("Sources")
                    tabs = st.tabs([f"Section {d['metadata'].get('id', '?')}" for d in sources])
                    
                    for tab, doc in zip(tabs, sources):
                        with tab:
                            meta = doc["metadata"]
                            st.markdown(f"**{meta.get('title', 'Untitled')}** (p. {meta.get('page_start', '?')})")
                            st.text(doc['content'])
                    
                    # Download Report
                    report = f"Query: {query}\nDate: {datetime.now()}\n\nAnswer:\n{answer}\n\nSources:\n"
                    for d in sources:
                        report += f"\n--- Section {d['metadata'].get('id')} ---\n{d['content']}\n"

                    st.download_button("Download Report", report, "legal_search_report.txt")

                # Update history
                st.session_state.history.append((
                    query,
                    answer,
                    datetime.now().strftime("%H:%M")
                ))
            else:
                st.error("Failed to generate response. Please try again.")

def main():
    init_session()
    app = LegalSearchApp()
    context = render_sidebar(app)
    render_main(app, context)
    
    st.divider()
    st.caption("Disclaimer: For educational purposes only. Consult a qualified lawyer for legal advice.")

if __name__ == "__main__":
    main()
