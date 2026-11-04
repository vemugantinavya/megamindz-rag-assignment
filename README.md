# Cross-Lingual Hindi RAG Engine

## 1. How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Make sure Tesseract-OCR and Poppler are installed on your system.
3. Add your `GEMINI_API_KEY` to a `.env` file in the root directory.
4. Run: `python main.py`

## 2. Chunking Strategy
I used LangChain's `RecursiveCharacterTextSplitter` with a chunk size of 600 and an overlap of 100. Crucially, I added the Hindi Danda ("।") as a primary separator so that sentences would not be split in the middle. This ensures the chunks contain complete thoughts.

## 3. Database Choice
I chose **ChromaDB** because it is lightweight, open-source, and runs locally without needing a Docker setup. It also easily allows me to store metadata (page number, chunk ID) alongside the embeddings, which was essential for the citation requirement.

## 4. Handling Hindi Text
This was the biggest challenge. Standard PDF extractors (like `pypdf`) failed on this document and returned broken text (missing matras). To fix this, I built an OCR pipeline using **EasyOCR** with the Hindi language pack. I converted the PDF pages to images using Poppler, ran OCR, and then normalized the text using Unicode NFC to ensure consistency. For embeddings, I used `BAAI/bge-m3`, a highly capable multilingual model.

## 5. Future Improvements
If I had more time, I would improve the metadata extraction to pull the actual section headings (currently marked as "Unknown"). I would also implement Hybrid Search (BM25 + Vector) to better handle specific acronyms like "SLV-III", and add a simple Streamlit UI.
