"""
app.py
======

Stage 3 of the pipeline: the user-facing Streamlit application.

This is a Retrieval-Augmented Generation (RAG) interface over the Canadian
Criminal Code. The flow for a typical query is:

  1. Embed the user's question and find the most relevant Code sections in the
     pre-built FAISS index (the "retrieval" half — see build_index.py).
  2. Feed those sections to Google Gemini as grounding context and ask it to
     answer the question, citing sections (the "generation" half).
  3. Show the AI answer plus the underlying source sections so the user can
     verify the claims against the primary text.

Users can also upload their own ``.txt`` document to use as context instead of
the Code index — useful for asking questions about a specific contract or
memo.

Requires a ``GOOGLE_API_KEY`` (Gemini) in the environment / ``.env`` and a
FAISS index already built via ``build_index.py``.

Pipeline position: ``pdf_scraper.py`` -> ``build_index.py`` -> ``app.py``.
"""

import streamlit as st
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load secrets from .env (GOOGLE_API_KEY) before anything tries to read them.
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config must be the first Streamlit call in the script. "wide" gives the
# source-text tabs room to breathe; the sidebar starts expanded so the upload
# and history controls are immediately visible.
st.set_page_config(
    page_title="Canadian Criminal Code Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


class LegalSearchApp:
    """Encapsulates the search + generation backend.

    A note on the ``_self`` parameter naming below: Streamlit's ``@st.cache_*``
    decorators try to hash a function's arguments to build a cache key. ``self``
    (a live app instance holding model handles) isn't hashable in a useful way,
    so by convention you prefix it with an underscore — ``_self`` — which tells
    Streamlit to *skip* hashing that argument. The cache key then depends only
    on the real inputs (the query and ``k``).
    """

    def __init__(self):
        self._setup_api()

    def _setup_api(self):
        """Build the Gemini client, failing fast if the key is missing.

        We stop the whole app (``st.stop``) rather than limp along, because
        every meaningful action depends on the generation API — a clear early
        error is far better than a confusing failure deep inside a query.

        Uses the modern ``google-genai`` SDK: a single ``Client`` object (held
        on the instance) is the entry point for all requests, replacing the
        deprecated module-level ``genai.configure()`` + ``GenerativeModel``
        pattern from the legacy ``google-generativeai`` package. Constructing
        the client does no network I/O — the key is validated on first request.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY not found in environment variables")
            st.stop()
        self.client = genai.Client(api_key=api_key)

    @st.cache_resource
    def load_models(_self):
        """Load the embedding model and FAISS index, cached for the session.

        ``@st.cache_resource`` is the right decorator for heavy, unhashable,
        reusable objects (ML models, DB connections): the model is loaded once
        per server process and shared across reruns and users, instead of being
        re-loaded on every keystroke.

        The embedding model here MUST match the one used in build_index.py
        (all-MiniLM-L6-v2) — querying an index with a different embedding space
        produces meaningless results.
        """
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

                # allow_dangerous_deserialization=True is required because FAISS
                # stores its docstore via pickle, which LangChain refuses to
                # load unless you opt in. This is safe ONLY because we built the
                # index ourselves; never enable it for an index from an
                # untrusted source (pickle can execute arbitrary code on load).
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
        """Return the ``k`` Code sections most relevant to ``query``.

        ``@st.cache_data`` memoises the *result* keyed on (query, k), so
        repeating a search — or re-running after an unrelated widget change —
        is instant and doesn't re-hit the index. We return plain dicts (not
        LangChain Document objects) so the cached value is simple, picklable
        data.

        Returns an empty list on failure; the UI treats that as "no results".
        """
        try:
            db = _self.load_models()

            # similarity_search embeds the query and returns the k nearest
            # Documents directly. (Earlier versions of this code used
            # ``as_retriever().get_relevant_documents()``, which is deprecated
            # in current LangChain — this call is the supported equivalent.)
            docs = db.similarity_search(query, k=k)

            return [{
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in docs]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def generate_answer(self, query: str, context: str) -> Optional[str]:
        """Ask Gemini to answer ``query`` using only the supplied ``context``.

        The prompt is the core of the RAG "guardrail": we instruct the model to
        ground its answer in the provided legal text, cite sections, admit when
        the context is insufficient, and append a not-legal-advice disclaimer.
        Grounding the model in retrieved text is what keeps it from inventing
        plausible-sounding but fictional provisions.

        Returns the generated text, or None if the API call fails (the caller
        surfaces a retry message).
        """
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
            # gemini-1.5-flash is chosen for speed/cost: RAG answers are short
            # and grounded, so the larger "pro" model buys little here. In the
            # google-genai SDK the model is named per-request (no "models/"
            # prefix and no separate GenerativeModel object); ``contents`` takes
            # the prompt string directly.
            response = self.client.models.generate_content(
                model="gemini-1.5-flash-001",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    def process_upload(self, uploaded_file) -> Optional[str]:
        """Read an uploaded ``.txt`` file into a string for use as context.

        We warn (but do not reject) very short files: a few dozen characters
        rarely give Gemini enough to work with, yet a user might legitimately
        want to ask about a short clause, so the decision is left to them.
        Returns None only when the file genuinely can't be decoded as UTF-8.
        """
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
    """Initialise per-session state.

    Streamlit reruns the whole script on every interaction, so anything that
    must survive between reruns lives in ``st.session_state``. Here we seed the
    query ``history`` list exactly once.
    """
    if "history" not in st.session_state:
        st.session_state.history = []


def render_sidebar(app: LegalSearchApp) -> Optional[str]:
    """Render the sidebar (file upload + query history) and return the uploaded
    context text, if any.

    The returned string, when present, signals ``render_main`` to answer from
    the uploaded document instead of searching the Code index.
    """
    with st.sidebar:
        st.header("Settings")

        uploaded_file = st.file_uploader(
            "Context Analysis",
            type=["txt"],
            help="Upload a specific legal document to analyze."
        )

        # If a file was uploaded, read it now so the success/length feedback
        # appears alongside the uploader rather than down in the main panel.
        context_text = None
        if uploaded_file:
            context_text = app.process_upload(uploaded_file)
            if context_text:
                st.success(f"Loaded {len(context_text)} chars")

        st.divider()
        st.subheader("Recent Queries")

        # Show the five most recent queries (newest last). Each history entry is
        # a (query, answer, timestamp) tuple; we only display the query and time
        # here — the stored answer is kept for the downloadable report and
        # potential future "re-open this result" features.
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
    """Render the main panel: the search box, the AI answer, and sources."""
    st.title("Canadian Criminal Code Search")
    st.caption("Search the Criminal Code and get AI-summarized answers with citations.")
    st.divider()

    query = st.text_input("Search Query", placeholder="e.g., What is the penalty for fraud over $5000?")

    # Only do work once there's a non-whitespace query. Streamlit reruns on
    # every keystroke, so this guard also stops us from querying on an empty box.
    if query and query.strip():
        with st.spinner("Analyzing..."):
            # Two grounding modes:
            #  - If the user uploaded a document, answer from that text alone
            #    (no retrieval, so there are no Code "sources" to show).
            #  - Otherwise, retrieve the most relevant Code sections and join
            #    their text into the context block.
            if context_text:
                context = context_text
                sources = []
                st.info("Using uploaded context.")
            else:
                sources = app.search_documents(query)
                context = "\n\n".join([d["content"] for d in sources])

            # If retrieval found nothing usable, bail out before spending a
            # Gemini call on empty context.
            if not context.strip():
                st.warning("No relevant information found.")
                return

            answer = app.generate_answer(query, context)

            if answer:
                st.subheader("Summary")
                st.info(answer)

                # Show the supporting sections (Code mode only) so the user can
                # check the AI's answer against the primary text. One tab per
                # source, labelled by section id.
                if sources:
                    st.subheader("Sources")
                    tabs = st.tabs([f"Section {d['metadata'].get('id', '?')}" for d in sources])

                    for tab, doc in zip(tabs, sources):
                        with tab:
                            meta = doc["metadata"]
                            st.markdown(f"**{meta.get('title', 'Untitled')}** (p. {meta.get('page_start', '?')})")
                            # section_type comes from build_index._get_category;
                            # display it in human-readable form ("public_order"
                            # -> "Public Order") so users see the topical area.
                            section_type = meta.get("section_type")
                            if section_type:
                                st.caption(f"Category: {section_type.replace('_', ' ').title()}")
                            st.text(doc['content'])

                    # Build a plain-text report bundling the query, answer and
                    # full source text, and offer it as a download for records.
                    report = f"Query: {query}\nDate: {datetime.now()}\n\nAnswer:\n{answer}\n\nSources:\n"
                    for d in sources:
                        report += f"\n--- Section {d['metadata'].get('id')} ---\n{d['content']}\n"

                    st.download_button("Download Report", report, "legal_search_report.txt")

                # Record this query in session history (query, answer, time).
                st.session_state.history.append((
                    query,
                    answer,
                    datetime.now().strftime("%H:%M")
                ))
            else:
                st.error("Failed to generate response. Please try again.")


def main():
    """Application entry point wiring the session, sidebar and main panel."""
    init_session()
    app = LegalSearchApp()
    context = render_sidebar(app)
    render_main(app, context)

    st.divider()
    st.caption("Disclaimer: For educational purposes only. Consult a qualified lawyer for legal advice.")


if __name__ == "__main__":
    main()
