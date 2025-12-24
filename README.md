# Canadian Criminal Code Search

A RAG-based search tool for the Canadian Criminal Code using Google Gemini and FAISS.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Create a `.env` file:
   ```bash
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

3. **Data:**
   Ensure `data/C-46.pdf` exists (Download from [Justice Laws Website](https://laws-lois.justice.gc.ca/eng/acts/C-46/)).

## Usage

1. **Extract Sections:**
   ```bash
   python pdf_scraper.py
   ```

2. **Build Index:**
   ```bash
   python build_index.py
   ```

3. **Run App:**
   ```bash
   streamlit run app.py
   ```

## Stack

*   **Streamlit:** UI
*   **LangChain:** Document processing
*   **FAISS:** Vector database
*   **Google Gemini:** Answer generation

## Disclaimer

This tool is for educational purposes only. It is not legal advice. Always consult a qualified lawyer.
