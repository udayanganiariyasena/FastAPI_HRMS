import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
import logging
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="HR Document RAG System",
    description="Retrieval-Augmented Generation for HR Documents",
    version="1.0"
)

class Question(BaseModel):
    question: str

def initialize_rag():
    try:
        # Check for required files and environment variables
        if not os.path.exists("hr_document.pdf"):
            raise FileNotFoundError("HR document not found. Please place 'hr_document.pdf' in the working directory.")
            
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        logger.info("Loading HR document...")
        loader = PyPDFLoader("hr_document.pdf")
        documents = loader.load()
        
        if not documents:
            raise ValueError("No documents were loaded from the PDF")
            
        logger.info(f"Loaded {len(documents)} pages from the document")
        
        # Split document into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n",
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(texts)} text chunks")
        
        # Initialize embeddings
        logger.info("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create FAISS vector store
        logger.info("Creating vector store...")
        db = FAISS.from_documents(
            documents=texts,
            embedding=embeddings,
            distance_strategy="METRIC_L2"
        )
        
        # Create QA chain
        logger.info("Initializing QA chain...")
        return RetrievalQA.from_chain_type(
            llm=OpenAI(
                temperature=0.7,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model_name="gpt-3.5-turbo-instruct"
            ),
            chain_type="stuff",
            retriever=db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=False
        )
        
    except Exception as e:
        logger.error(f"RAG initialization failed: {str(e)}")
        return None

# Initialize RAG system at startup
rag_qa = initialize_rag()

@app.get("/")
async def health_check():
    return {
        "status": "running",
        "rag_ready": rag_qa is not None,
        "message": "RAG system is not ready" if rag_qa is None else "RAG system is ready"
    }

@app.post("/ask")
async def ask_question(question: Question):
    if not rag_qa:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized properly. Check logs for details."
        )
    
    try:
        result = rag_qa({"query": question.question})
        return {
            "response": result["result"],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
