import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel




from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment variables")

MODEL_NAME = os.getenv("GROQ_MODEL", "llama3-8b-8192")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
FAISS_DIR = DATA_DIR / "faiss_index"
STATIC_INDEX = Path("static/index.html")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}

app = FastAPI(title="RAG Application")

# Process-wide state.
embeddings = None
vector_store = None
store_lock = asyncio.Lock()

# Note: this memory is shared across all users of the process.
# For a real multi-user app, move memory/chat history to per-user session storage.
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)
query_cache: dict[str, dict[str, Any]] = {}


class QueryRequest(BaseModel):
    question: str
    k: int = 3


def get_embeddings() -> FastEmbedEmbeddings:
    global embeddings
    if embeddings is None:
        logger.info("Loading FastEmbed embeddings...")
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embeddings


def get_vector_store():
    global vector_store
    if vector_store is not None:
        return vector_store

    index_file = FAISS_DIR / "index.faiss"
    meta_file = FAISS_DIR / "index.pkl"

    if index_file.exists() and meta_file.exists():
        logger.info("Loading existing FAISS index...")
        vector_store = FAISS.load_local(
            str(FAISS_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    else:
        vector_store = None

    return vector_store


def load_document(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return loader.load()


def build_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No readable text could be extracted from the document.")
    return FAISS.from_documents(chunks, get_embeddings())


def process_uploaded_file(file_path: Path) -> None:
    global vector_store

    logger.info("STEP 1: Loading document...")
    docs = load_document(str(file_path))

    logger.info("STEP 2: Splitting and embedding...")
    new_store = build_vector_store(docs)

    logger.info("STEP 3: Merging vector store...")
    current_store = get_vector_store()

    if current_store is None:
        vector_store = new_store
    else:
        current_store.merge_from(new_store)
        vector_store = current_store

    logger.info("STEP 4: Saving FAISS index...")
    vector_store.save_local(str(FAISS_DIR))

    query_cache.clear()
    logger.info("SUCCESS: %s indexed", file_path.name)


def save_uploaded_file(upload: UploadFile, destination: Path) -> None:
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)


def get_qa_chain(store, k: int = 3):
    llm = ChatGroq(
        model_name=MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
    )

    prompt_template = """Use the following pieces of context to answer the question at the end.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Helpful Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": max(1, min(k, 10))},
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )


def invoke_chain(chain, question: str):
    return chain.invoke({"question": question}, config={"timeout": 20})


@app.get("/", response_class=HTMLResponse)
async def home():
    if STATIC_INDEX.exists():
        return STATIC_INDEX.read_text(encoding="utf-8")
    return "<h1>Error: static/index.html not found. Check your folder structure.</h1>"


@app.get("/documents")
async def list_documents():
    files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    return {"documents": sorted(files)}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning("Failed upload attempt: unsupported file type '%s'", ext)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    file_path = UPLOAD_DIR / filename

    try:
        await run_in_threadpool(save_uploaded_file, file, file_path)

        async with store_lock:
            await run_in_threadpool(process_uploaded_file, file_path)

        return {"message": f"Successfully uploaded and indexed '{filename}'."}

    except Exception as e:
        logger.exception("Error processing upload for %s", filename)
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        await file.close()


@app.post("/query")
async def query_document(req: QueryRequest):
    store = get_vector_store()

    if store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload a document first.",
        )

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    cache_key = f"{question.lower()}::k={max(1, min(req.k, 10))}"
    if cache_key in query_cache:
        logger.info("Serving query from cache.")
        return query_cache[cache_key]

    try:
        logger.info("Processing query: %s", question)
        qa_chain = get_qa_chain(store, k=req.k)
        res = await run_in_threadpool(invoke_chain, qa_chain, question)

        answer = res.get("answer", "")
        source_docs = res.get("source_documents", [])

        sources = [{"content": doc.page_content.strip()} for doc in source_docs]

        payload = {
            "answer": answer,
            "sources": sources,
        }

        query_cache[cache_key] = payload
        return payload

    except Exception as e:
        logger.exception("Error during query processing")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_stream")
async def stream_query_document(req: QueryRequest):
    store = get_vector_store()

    if store is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    llm_streaming = ChatGroq(
        model_name=MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
        streaming=True,
    )

    prompt_template = """Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": max(1, min(req.k, 10))},
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_streaming,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    async def generate_response():
        logger.info("Streaming response started...")
        async for chunk in chain.astream({"question": question}):
            if "answer" in chunk:
                yield chunk["answer"]

    return StreamingResponse(generate_response(), media_type="text/plain")


@app.delete("/history")
async def clear_memory():
    memory.clear()
    query_cache.clear()
    logger.info("Cleared conversation memory and cache.")
    return {"message": "Memory and cache cleared."}
