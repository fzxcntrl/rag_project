import os
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
app = FastAPI(title="RAG Application")
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not set in environment variables")

# --------------- Upgrade 12: Structured Logging ---------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --------------- state ---------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
FAISS_DIR = "faiss_index"

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
embeddings = None
vector_store = None

# Upgrade 6 & 9: Conversation Memory & Query Caching
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    k=5
)
query_cache: dict = {}

def get_embeddings():
    """Lazy load embeddings to save memory on startup."""
    global embeddings
    if embeddings is None:
        logger.info("Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def get_vector_store():
    """Lazy load or retrieve the FAISS vector store."""
    global vector_store
    if vector_store is None:
        if Path(FAISS_DIR).exists():
            logger.info("Loading existing FAISS index...")
            vector_store = FAISS.load_local(
                FAISS_DIR,
                get_embeddings(),
                allow_dangerous_deserialization=True
            )
        else:
            logger.info("No FAISS index found.")
            vector_store = None
    return vector_store

# --------------- helpers ---------------
def load_document(file_path: str):
    """Load a PDF or text file and return LangChain Documents."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return loader.load()

def build_vector_store(documents):
    """Split documents into chunks and build a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    store = FAISS.from_documents(chunks, get_embeddings())
    return store

def get_qa_chain(store, k: int = 3):
    """Create a ConversationalRetrievalQA chain from the vector store."""
    llm = ChatGroq(model_name="llama3-8b-8192",groq_api_key=os.getenv("GROQ_API_KEY"))
    
    # Upgrade 1: Prompt Templating
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If the answer is not in the context, just say "I don't know based on the provided documents", don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    Helpful Answer:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Upgrade 11 & 4: Configurable Retrieval & Similarity Score Threshold
    retriever = store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": 0.3} # Lowered threshold slightly for better recall
    )
    
    # Upgrade 6 & 3: Memory and Source Chunks
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return qa_chain


# --------------- routes ---------------
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the index.html from the static folder."""
    index_path = Path("static/index.html")
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>Error: static/index.html not found. Check your folder structure.</h1>"


# Upgrade 10: /documents Endpoint
@app.get("/documents")
async def list_documents():
    """Returns a list of all uploaded files."""
    files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    return {"documents": sorted(files)}


class QueryRequest(BaseModel):
    question: str
    k: int = 3  # Upgrade 4: Configurable Top-k


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store
    
    # Input Validation Guard
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Failed upload attempt: Unsupported file type '{ext}'")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        docs = load_document(str(file_path))
        new_store = build_vector_store(docs)
        
        current_store = get_vector_store()
        if current_store is None:
            vector_store = new_store
        else:
            current_store.merge_from(new_store)
            vector_store = current_store
            
        # Upgrade 2: Persist the FAISS Index after merging
        vector_store.save_local(FAISS_DIR)
        logger.info(f"Successfully processed and indexed {file.filename}")
            
        return {"message": f"Successfully uploaded and indexed '{file.filename}'."}
        
    except Exception as e:
        logger.exception(f"Error processing upload for {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_document(req: QueryRequest):
    store = get_vector_store()

    if store is None:  
        raise HTTPException(
            status_code=400, 
            detail="No documents uploaded yet. Please upload a document first."
        )
        
    # Upgrade 9: Query Caching
    cache_key = req.question.strip().lower()
    if cache_key in query_cache:
        logger.info("Serving query from cache.")
        return query_cache[cache_key]
        
    qa_chain = get_qa_chain(store, k=req.k)
    
    try:
        logger.info(f"Processing query: {req.question}")
        res = qa_chain.invoke({"question": req.question},config={"timeout": 20}

        answer = res.get("answer", "")
        source_docs = res.get("source_documents", [])
        
        sources = [{"content": doc.page_content.strip()} for doc in source_docs]
        
        result_payload = {
            "answer": answer,
            "sources": sources
        }
        
        # Save to cache
        query_cache[cache_key] = result_payload
        return result_payload
        
    except Exception as e:
        logger.exception("Error during query processing")
        raise HTTPException(status_code=500, detail=str(e))


# Upgrade 7: Streaming Responses (Alternative Endpoint)
@app.post("/query_stream")
async def stream_query_document(req: QueryRequest):
    """
    Alternative endpoint that streams the LLM response token by token.
    Note: Calling this endpoint will return raw text chunks instead of JSON.
    """
    store = get_vector_store()
    if store is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    
    # Reinitialize a chain specifically for streaming
    llm_streaming = ChatGroq(model_name="llama3-8b-8192", streaming=True)
    retriever = store.as_retriever(search_kwargs={"k": req.k})
    
    prompt_template = """Use the following context to answer the question. \n\nContext: {context}\n\nQuestion: {question}\nAnswer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_streaming, 
        retriever=retriever, 
        memory=memory, 
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    async def generate_response():
        logger.info("Streaming response started...")
        async for chunk in chain.astream({"question": req.question}):
            # Yield answer tokens as they arrive
            if "answer" in chunk:
                yield chunk["answer"]

    return StreamingResponse(generate_response(), media_type="text/plain")


@app.delete("/history")
async def clear_memory():
    """Clears the conversational memory and the query cache."""
    memory.clear()
    query_cache.clear()
    logger.info("Cleared conversation memory and cache.")
    return {"message": "Memory and cache cleared."}