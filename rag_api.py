import os
import re
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed
from bangla_pdf_ocr import process_pdf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Disable TensorFlow GPU warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load environment variables
load_dotenv()

# Configuration
PDF_PATH = "dataset\HSC26-Bangla1st-Paper.pdf"
TEXT_CACHE_PATH = "dataset\paper_extracted.txt"
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Pydantic model for request body
class QueryRequest(BaseModel):
    query: str
    expected_answer: str | None = None

# FastAPI app
app = FastAPI(title="Bangla RAG API", description="API for querying Bangla literature using a RAG system", version="1.0.0")

# Load and preprocess PDF with bangla_pdf_ocr, with caching
def load_and_preprocess_pdf(pdf_path: str, cache_path: str):
    try:
        # Check if cached text file exists
        if os.path.exists(cache_path):
            logger.info(f"Loading cached text from: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"Loaded cached text sample: {text[:500]}")
        else:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            logger.info(f"Extracting text from PDF: {pdf_path}")
            text = process_pdf(pdf_path)
            if not text or not text.strip():
                raise ValueError("No text extracted from PDF with bangla_pdf_ocr")

            logger.info(f"Extracted text sample: {text[:500]}")
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved extracted text to: {cache_path}")

        # Clean the extracted text
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        cleaned_text = cleaned_text.encode('utf-8', errors='replace').decode('utf-8')
        logger.info(f"Cleaned text sample: {cleaned_text[:500]}")

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
        logger.info("Initializing HuggingFace embeddings")
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

# Create or load FAISS vector store
def get_vector_store():
    try:
        embeddings = initialize_embeddings()
        if os.path.exists(FAISS_INDEX_PATH):
            logger.info(f"Loading FAISS vector store from: {FAISS_INDEX_PATH}")
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS vector store loaded successfully")
        else:
            logger.info("FAISS index not found, creating new vector store")
            docs = load_and_preprocess_pdf(PDF_PATH, TEXT_CACHE_PATH)
            vector_store = FAISS.from_documents(docs, embeddings)
            vector_store.save_local(FAISS_INDEX_PATH)
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
            model_name='llama-3.3-70b-versatile', #"llama3-70b-8192", 
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

# Global RAG chain (initialized once)
try:
    vector_store = get_vector_store()
    rag_chain = create_rag_chain(vector_store)
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    raise

# API endpoint for querying
@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing query: {request.query}")
        retrieved_docs = rag_chain.retriever.invoke(request.query)
        context = [f"[Doc {i+1}]: {doc.page_content[:500]}" for i, doc in enumerate(retrieved_docs)]

        result = rag_chain.invoke({"query": request.query})
        answer = result.get("result", "").strip()

        response = {
            "query": request.query,
            "answer": answer,
            "context": context
        }

        if request.expected_answer:
            embeddings = initialize_embeddings()
            answer_embedding = embeddings.embed_query(answer if answer else "উত্তর পাওয়া যায়নি")
            expected_embedding = embeddings.embed_query(request.expected_answer)
            sim = np.dot(answer_embedding, expected_embedding) / (
                np.linalg.norm(answer_embedding) * np.linalg.norm(expected_embedding)
            )
            response["expected_answer"] = request.expected_answer
            response["cosine_similarity"] = float(sim)

        return response
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

