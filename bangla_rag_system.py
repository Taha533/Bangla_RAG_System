import os
import re
import logging
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from bangla_pdf_ocr import process_pdf
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_fixed

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Disable TensorFlow GPU warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load environment variables
load_dotenv()

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PDF_PATH = "dataset\HSC26-Bangla1st-Paper.pdf"
TEXT_PATH = "dataset\HSC26-Bangla1st-Paper.txt"


# Load and preprocess PDF with bangla_pdf_ocr, with caching
def load_and_preprocess_pdf(pdf_path: str, cache_path: str):
    try:
        # Check if text file exists
        if os.path.exists(cache_path):
            logger.info(f"Loading text from: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            logger.info(f"Extracting text from PDF: {pdf_path}")
            # Extract text using bangla_pdf_ocr
            text = process_pdf(pdf_path)
            if not text or not text.strip():
                raise ValueError("No text extracted from PDF with bangla_pdf_ocr")
            logger.info(f"Extracted text sample: {text[:500]}")
            # Save extracted text to txt file
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved extracted text to: {cache_path}")

        # Clean the extracted text
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        cleaned_text = cleaned_text.encode('utf-8', errors='replace').decode('utf-8')
        # Create a single Document object
        data = [Document(page_content=cleaned_text, metadata={"source": pdf_path})]

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "।", " ", ""],
        )
        chunks = text_splitter.split_documents(data)
        logger.info(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i+1}: {chunk.page_content[:200]}")
        return chunks
    except Exception as e:
        logger.error(f"Error loading PDF or cache: {e}")
        raise

# HuggingFace embeddings for Bengali
def initialize_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="l3cube-pune/bengali-sentence-similarity-sbert",
            encode_kwargs={'normalize_embeddings': True}
        )
        _ = embeddings.embed_query("test")
        logger.info("HuggingFace embeddings initialized successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding init error: {e}")
        raise

# Create FAISS vector store
def create_vector_store(docs):
    try:
        logger.info("Creating FAISS vector store")
        embeddings = initialize_embeddings()
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local("./faiss_index")
        logger.info("FAISS vector store created and saved")
        return vector_store
    except Exception as e:
        logger.error(f"Vector store error: {e}")
        raise

# Initialize Groq LLM with rate limit handling
@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
def initialize_llm():
    try:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        logger.info("Initializing Groq LLM")
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=512,
        )
    except Exception as e:
        logger.error(f"LLM init error: {e}")
        raise

# Create RAG chain
def create_rag_chain(vector_store):
    try:
        logger.info("Creating RAG chain")
        llm = initialize_llm()
        prompt_template = """
        You are an expert in Bengali literature. Use the provided context to answer the question in Bengali. Extract the exact answer from the context if available, or say "উত্তর পাওয়া যায়নি" (Answer not found) if the context does not contain the answer. Provide only the answer, without repeating the question or context.
        Context: {context}
        Question: {question}
        Answer:
        """
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        logger.info("RAG chain created successfully")
        return chain
    except Exception as e:
        logger.error(f"RAG chain error: {e}")
        raise

# Evaluate RAG system
def evaluate_rag(query: str, expected: str, rag_chain):
    try:
        logger.info(f"Evaluating query: {query}")
        # retrieved_docs = rag_chain.retriever.invoke(query)
        result = rag_chain.invoke({"query": query})
        answer = result.get("result", "").strip()

        embeddings = initialize_embeddings()
        answer_embedding = embeddings.embed_query(answer if answer else "উত্তর পাওয়া যায়নি")
        expected_embedding = embeddings.embed_query(expected)
        sim = np.dot(answer_embedding, expected_embedding) / (
            np.linalg.norm(answer_embedding) * np.linalg.norm(expected_embedding)
        )
        return {
            "query": query,
            "expected": expected,
            "actual": answer,
            "cosine_similarity": float(sim)
        }
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {"error": str(e)}

# Main
def main():
    try:
        logger.info("Starting RAG system")
        docs = load_and_preprocess_pdf(PDF_PATH, TEXT_PATH)
        vector_store = create_vector_store(docs)
        rag_chain = create_rag_chain(vector_store)

        sample_queries = [
            {"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "expected": "শুম্ভুনাথ"},
            {"query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "expected": "মামাকে"},
            {"query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "expected": "১৫ বছর"}
        ]

        for s in sample_queries:
            result = evaluate_rag(s["query"], s["expected"], rag_chain)
            print(f"\nQuery: {s['query']}\nExpected: {s['expected']}\nActual: {result.get('actual', 'N/A')}\nCosine Similarity: {result.get('cosine_similarity', 'N/A')}")
    except Exception as e:
        logger.error(f"Main failed: {e}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
