# import os
# import logging
# from dotenv import load_dotenv
# from langchain_community.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from tenacity import retry, stop_after_attempt, wait_fixed

# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# # Disable TensorFlow GPU warnings
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# # Load environment variables
# load_dotenv()

# # Configuration
# FAISS_INDEX_PATH = "./faiss_index"

# # Initialize HuggingFace embeddings for Bengali
# def initialize_embeddings():
#     try:
#         embeddings = HuggingFaceEmbeddings(
#             model_name="l3cube-pune/bengali-sentence-similarity-sbert",
#             encode_kwargs={'normalize_embeddings': True}
#         )
#         _ = embeddings.embed_query("test")
#         logger.info("HuggingFace embeddings initialized successfully")
#         return embeddings
#     except Exception as e:
#         logger.error(f"Embedding init error: {e}")
#         raise

# # Load FAISS vector store
# def load_vector_store():
#     try:
#         logger.info("Loading FAISS vector store")
#         embeddings = initialize_embeddings()
#         if not os.path.exists(FAISS_INDEX_PATH):
#             raise FileNotFoundError(f"FAISS index not found at: {FAISS_INDEX_PATH}")
#         vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
#         logger.info("FAISS vector store loaded successfully")
#         return vector_store
#     except Exception as e:
#         logger.error(f"Vector store load error: {e}")
#         raise

# # Initialize Groq LLM with rate limit handling
# @retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
# def initialize_llm():
#     try:
#         GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#         if not GROQ_API_KEY:
#             raise ValueError("GROQ_API_KEY not found in environment variables")
#         logger.info("Initializing Groq LLM")
#         return ChatGroq(
#             groq_api_key=GROQ_API_KEY,
#             model_name="llama3-70b-8192",
#             temperature=0.7,
#             max_tokens=512,
#         )
#     except Exception as e:
#         logger.error(f"LLM init error: {e}")
#         raise

# # Create RAG chain
# def create_rag_chain(vector_store):
#     try:
#         logger.info("Creating RAG chain")
#         llm = initialize_llm()
#         prompt_template = """
#         You are an expert in Bengali literature. Use the provided context to answer the question in Bengali extractively. Extract a short and exact answer from the context if available, or say "উত্তর পাওয়া যায়নি" (Answer not found) if the context does not contain the answer. Provide only the answer, without repeating the question or context.
#         Context: {context}
#         Question: {question}
#         Answer:
#         """
#         PROMPT = PromptTemplate(
#             template=prompt_template,
#             input_variables=["context", "question"]
#         )
#         chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
#             chain_type_kwargs={"prompt": PROMPT}
#         )
#         logger.info("RAG chain created successfully")
#         return chain
#     except Exception as e:
#         logger.error(f"RAG chain error: {e}")
#         raise

# # Run inference
# def run_inference(rag_chain, query: str):
#     try:
#         logger.info(f"Processing query: {query}")
#         # # Get retrieved context
#         # retrieved_docs = rag_chain.retriever.invoke(query)

#         # Run the chain
#         result = rag_chain.invoke({"query": query})
#         answer = result.get("result", "").strip()
#         # logger.info(f"Answer: {answer}")
#         return answer
#     except Exception as e:
#         logger.error(f"Inference error: {e}")
#         return f"Error: {str(e)}"

# # Main
# def main():
#     try:
#         logger.info("Starting RAG inference system")
#         vector_store = load_vector_store()
#         rag_chain = create_rag_chain(vector_store)

#         print("\nRAG Inference System Ready!")
#         print("Enter your query in Bengali (or type 'exit' to quit):\n")

#         while True:
#             query = input("> ")
#             if query.lower() == "exit":
#                 logger.info("Exiting inference system")
#                 break
#             if not query.strip():
#                 print("Please enter a valid query.")
#                 continue
#             answer = run_inference(rag_chain, query)
#             print(f"\nAnswer: {answer}\n")
#     except Exception as e:
#         logger.error(f"Inference system failed: {e}")
#         print(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()



import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_fixed

# Disable TensorFlow GPU warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load environment variables
load_dotenv()

# Configuration
FAISS_INDEX_PATH = "./faiss_index"

# Initialize HuggingFace embeddings for Bengali
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name="l3cube-pune/bengali-sentence-similarity-sbert",
        encode_kwargs={'normalize_embeddings': True}
    )

# Load FAISS vector store
def load_vector_store():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at: {FAISS_INDEX_PATH}")
    embeddings = initialize_embeddings()
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Initialize Groq LLM with rate limit handling
@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
def initialize_llm():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=512,
    )

# Create RAG chain
def create_rag_chain(vector_store):
    llm = initialize_llm()
    prompt_template = """
    You are an expert in Bengali literature. Use the provided context to answer the question in Bengali extractively. Extract a short and exact answer from the context if available, or say "উত্তর পাওয়া যায়নি" (Answer not found) if the context does not contain the answer. Provide only the answer, without repeating the question or context.
    Context: {context}
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT}
    )

# Run inference
def run_inference(rag_chain, query: str):
    try:
        result = rag_chain.invoke({"query": query})
        return result.get("result", "").strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Main
def main():
    try:
        vector_store = load_vector_store()
        rag_chain = create_rag_chain(vector_store)

        print("\nRAG Inference System Ready!")
        print("Enter your query in Bengali (or type 'exit' to quit):\n")

        while True:
            query = input("> ")
            if query.lower() == "exit":
                break
            if not query.strip():
                print("Please enter a valid query.")
                continue
            answer = run_inference(rag_chain, query)
            print(f"\nপ্রশ্ন: {query}\nউত্তর: {answer}\n")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()