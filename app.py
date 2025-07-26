import os
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import numpy as np
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
FAISS_INDEX_PATH = "faiss_index"
TEXT_PATH = "dataset\HSC26-Bangla1st-Paper.txt"

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

# Load FAISS vector store
def load_vector_store():
    try:
        logger.info(f"Loading FAISS vector store from: {FAISS_INDEX_PATH}")
        embeddings = initialize_embeddings()
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at: {FAISS_INDEX_PATH}")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS vector store loaded successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Vector store loading error: {e}")
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

# Evaluate RAG system with optional expected answer and chat history update
def evaluate_rag(query: str, expected: str = None, rag_chain=None):
    try:
        logger.info(f"Evaluating query: {query}")
        retrieved_docs = rag_chain.retriever.invoke(query)
        context = "\n".join([f"[Doc {i+1}]: {doc.page_content[:500]}" for i, doc in enumerate(retrieved_docs)])

        result = rag_chain.invoke({"query": query})
        answer = result.get("result", "").strip()

        # Update chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"query": query, "answer": answer})

        response = {
            "query": query,
            "actual": answer,
            "context": context
        }

        if expected:
            embeddings = initialize_embeddings()
            answer_embedding = embeddings.embed_query(answer if answer else "উত্তর পাওয়া যায়নি")
            expected_embedding = embeddings.embed_query(expected)
            sim = np.dot(answer_embedding, expected_embedding) / (
                np.linalg.norm(answer_embedding) * np.linalg.norm(expected_embedding)
            )
            response["expected"] = expected
            response["cosine_similarity"] = float(sim)

        return response
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {"error": str(e)}

# Streamlit app
def main():
    st.set_page_config(page_title="Bengali RAG Q&A", page_icon="📚", layout="wide")
    
    # Sidebar for chat history
    with st.sidebar:
        st.header("Chat History")
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if st.session_state.chat_history:
            for i, entry in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Query {len(st.session_state.chat_history) - i}: {entry['query'][:50]}..."):
                    st.write(f"**Query**: {entry['query']}")
                    st.write(f"**Answer**: {entry['answer']}")
        else:
            st.write("No queries yet.")

    # Main content
    st.title("📚 Bengali Literature Q&A System")
    st.markdown("Ask questions in Bengali about the document, and the system will provide answers based on the context.")

    # Initialize session state for RAG chain
    if 'rag_chain' not in st.session_state:
        try:
            vector_store = load_vector_store()
            st.session_state.rag_chain = create_rag_chain(vector_store)
            st.success("RAG system initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {str(e)}")
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            return

    # Query input
    query = st.text_input("আপনার প্রশ্ন লিখুন (Enter your question in Bengali):", placeholder="উদাহরণ: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?")
    expected_answer = st.text_input("প্রত্যাশিত উত্তর (Optional expected answer, leave blank if none):", placeholder="উদাহরণ: শুম্ভুনাথ")
    show_context = st.checkbox("Show retrieved context", value=False)

    if st.button("Submit Query"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        with st.spinner("Processing your query..."):
            try:
                result = evaluate_rag(query, expected_answer if expected_answer.strip() else None, st.session_state.rag_chain)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.subheader("Result")
                    st.write(f"**Query**: {result['query']}")
                    st.write(f"**Answer**: {result['actual']}")
                    
                    if show_context:
                        st.subheader("Retrieved Context")
                        st.text(result['context'])
                    
                    if "expected" in result:
                        st.write(f"**Expected Answer**: {result['expected']}")
                        st.write(f"**Cosine Similarity**: {result['cosine_similarity']:.4f}")
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                logger.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()