# Canadian Criminal Code Search

A Retrieval-Augmented Generation (RAG) search tool for the Canadian Criminal
Code. Ask a plain-English question, and the app finds the most relevant
sections of the Code and uses Google Gemini to summarise an answer **with
citations back to the primary text**.

> ⚖️ **Disclaimer:** This tool is for educational purposes only. It is **not**
> legal advice. Always consult a qualified lawyer.

---

## How it works

This is a classic two-phase RAG system. The text is indexed once, offline; then
every query retrieves relevant text and grounds the language model in it.

```
   data/C-46.pdf
        │
        ▼
  pdf_scraper.py        Stage 1 — Extract
        │   parses the PDF into individual numbered sections
        ▼
  data/processed/sections.json
        │
        ▼
  build_index.py        Stage 2 — Index
        │   embeds each section locally (all-MiniLM-L6-v2)
        ▼
  faiss_index/          (index.faiss + index.pkl)
        │
        ▼
  app.py                Stage 3 — Search & Answer
            • embeds your question and finds the nearest sections (FAISS)
            • sends those sections to Gemini as grounding context
            • shows the answer + the source sections it was based on
```

**Why RAG?** A language model on its own will happily invent plausible-sounding
but fictional legal provisions. By retrieving the actual Code text first and
instructing the model to answer *only* from that text (and to cite sections),
the answers stay grounded in the primary source — and you can verify them
against the "Sources" tabs in the UI.

---

## Project layout

| Path | Purpose |
|------|---------|
| `pdf_scraper.py` | Stage 1 — extracts numbered sections from the PDF into `sections.json`. |
| `build_index.py` | Stage 2 — embeds the sections and builds the FAISS vector index. |
| `app.py` | Stage 3 — the Streamlit UI: retrieval + Gemini answer generation. |
| `data/C-46.pdf` | The source Criminal Code PDF (you supply this — see Setup). |
| `data/processed/sections.json` | Extracted, cleaned section records. |
| `faiss_index/` | The persisted vector index (`index.faiss`, `index.pkl`). |
| `tests/test_data.py` | Smoke tests verifying the data artefacts exist and parse. |

---

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your API key**

   The answer-generation step uses Google Gemini, which needs an API key. Get
   one from [Google AI Studio](https://aistudio.google.com/app/apikey), then
   create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
   (`.env` is git-ignored, so your key won't be committed.)

3. **Supply the source PDF**

   Download the Criminal Code (`C-46.pdf`) from the
   [Justice Laws Website](https://laws-lois.justice.gc.ca/eng/acts/C-46/) and
   place it at `data/C-46.pdf`.

---

## Usage

Run the three stages in order. Stages 1 and 2 are one-time (re-run only when the
PDF changes); stage 3 is the app you interact with.

1. **Extract sections from the PDF**
   ```bash
   python pdf_scraper.py
   ```
   Writes `data/processed/sections.json`.

2. **Build the search index**
   ```bash
   python build_index.py
   ```
   Embeds the sections locally (no API key needed for this step) and writes the
   FAISS index to `faiss_index/`. The first run downloads the ~80 MB embedding
   model.

3. **Launch the app**
   ```bash
   streamlit run app.py
   ```
   Then open the URL Streamlit prints (usually <http://localhost:8501>).

### Using the app

- **Search the Code:** type a question (e.g. *"What is the penalty for fraud
  over $5000?"*). You'll get an AI summary plus a tab per source section, each
  showing the title, page, topical category, and full text.
- **Analyse your own document:** upload a `.txt` file in the sidebar to have the
  model answer from *that* text instead of the Code index.
- **History & export:** recent queries appear in the sidebar, and any
  Code-search result can be exported via **Download Report**.

---

## Running the tests

```bash
python -m unittest discover tests
```

These verify that the PDF, `sections.json`, and FAISS index are present and
well-formed — i.e. that you've run the pipeline before launching the app.

---

## Tech stack

| Component | Role |
|-----------|------|
| **Streamlit** | Web UI and session state. |
| **PyMuPDF (`fitz`)** | Fast PDF text extraction. |
| **LangChain** | Document/vector-store abstractions. |
| **`all-MiniLM-L6-v2`** | Local sentence-embedding model (CPU, no API key). |
| **FAISS** | Vector index for nearest-neighbour semantic search. |
| **Google Gemini** | Grounded answer generation. |

---

## Notes & limitations

- **Bilingual source text.** The official PDF interleaves English and French.
  Because extraction is linear, a section's stored text can contain both
  languages. Semantic search still matches English queries against the English
  wording, but raw source text may look duplicated/translated. Splitting the
  languages cleanly would require column-aware layout parsing.
- **Embedding model must match.** The index in `faiss_index/` is tied to the
  embedding model named in `build_index.py` (`all-MiniLM-L6-v2`). If you change
  that model, you must rebuild the index — querying with a mismatched model
  yields meaningless results.
- **Topical categories are coarse.** The "Category" shown per source is derived
  from section-number ranges and is for orientation only, not legal precision.

## Disclaimer

This tool is for educational purposes only. It is not legal advice. Always
consult a qualified lawyer.
