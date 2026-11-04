import easyocr
from pdf2image import convert_from_path
import unicodedata
import numpy as np
import os
import time
import json # <-- Added to save/load text
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- CONFIGURATION ---
load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel('gemini-3.6-flash')

POPPLER_PATH = r'C:\Users\vemug\Downloads\Release-26.09.0-0\poppler-26.09.0\Library\bin' 
PDF_PATH = "data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"
CACHE_FILE = "extracted_text.json" # <-- The file that saves our OCR progress

# --- 1. OCR & EXTRACTION (With Caching) ---
def extract_text_with_easyocr(pdf_path):
    # Check if we already did the OCR and saved it
    if os.path.exists(CACHE_FILE):
        print(f"✅ Found cached text file '{CACHE_FILE}'. Skipping OCR (Loading instantly)...")
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    print("Step 1: Running OCR on all pages. (This takes time, but will only happen ONCE)...")
    reader = easyocr.Reader(['hi', 'en']) 
    images = convert_from_path(pdf_path, dpi=150, poppler_path=POPPLER_PATH) 
    extracted_data = []

    for i, image in enumerate(images):
        image_np = np.array(image) 
        result = reader.readtext(image_np, detail=0, paragraph=True)
        text = " ".join(result)
        normalized_text = unicodedata.normalize('NFC', text)
        extracted_data.append({
            "page_number": i + 1,
            "text": normalized_text
        })
        print(f"Processed page {i+1}/{len(images)}")
        
    # Save the extracted data to a file so we never have to run OCR again
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        print(f"✅ Saved OCR text to '{CACHE_FILE}'")
        
    return extracted_data

# --- 2. CHUNKING & VECTOR DB ---
def build_vector_db(documents):
    print("Step 2: Chunking and Embedding...")
    docs = [Document(page_content=doc["text"], metadata={"page": doc["page_number"], "section": "Unknown"}) for doc in documents]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "।", ".", "\n", " "]
    )
    chunks = text_splitter.split_documents(docs)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

# --- 3. RAG PIPELINE ---
def answer_query(query, vectorstore):
    results = vectorstore.similarity_search_with_score(query, k=3)
    context_text = "\n\n".join([f"[Page {doc.metadata['page']}] {doc.page_content}" for doc, score in results])
    
    prompt = f"""
    You are a helpful assistant. Answer the user's question based ONLY on the provided context. 
    If the answer is not in the context, say "I don't know based on the document."
    Always answer in the same language as the query.
    
    Context:
    {context_text}
    
    Question: {query}
    """
    
    response = llm.generate_content(prompt)
    top_doc, top_score = results[0]
    citation = f"page: {top_doc.metadata['page']} · section: {top_doc.metadata['section']} · chunk_id: {top_doc.metadata['chunk_id']} · score: {top_score:.2f}"
    return response.text, citation

# --- 4. RUN THE 6 QUERIES ---
if __name__ == "__main__":
    raw_documents = extract_text_with_easyocr(PDF_PATH)
    db = build_vector_db(raw_documents)
    
    test_queries = [
        "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?",
        "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?",
        "पोखरण-II में कलाम की क्या भूमिका थी?",
        "Which institution did Kalam attend to study aeronautical engineering?",
        "Who co-wrote the autobiography, and in what year was it published?",
        "How and where did Kalam die in 2015?"
    ]
    
    print("\n" + "="*50)
    print("RUNNING TEST QUERIES")
    print("="*50 + "\n")
    
    for i, q in enumerate(test_queries):
        print(f"Query {i+1}: {q}")
        try:
            answer, citation = answer_query(q, db)
            print(f"Answer: {answer}")
            print(f"Source: {citation}\n")
        except Exception as e:
            print(f"Error on this query: {e}\n")
            
        print("-" * 50)
        
        if i < len(test_queries) - 1:
            print("Waiting 15 seconds for API rate limit...")
            time.sleep(15)