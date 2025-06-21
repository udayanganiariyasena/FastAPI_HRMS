from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

def initialize_rag():
    # Load PDF
    loader = PyPDFLoader("hr_document.pdf")
    documents = loader.load()
    
    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    
    # Create QA chain
    return RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.3),  # Lower temperature for more factual answers
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),  # Retrieve 4 most relevant chunks
        return_source_documents=False
    )

rag_qa = initialize_rag()

@app.get("/")
def read_root():
    return {"message": "AI HR Assistant"}

@app.post("/ask")
async def ask_question(question: Question):
    try:
        result = rag_qa({"query": question.question})
        return {
            "response": result["result"],
            "status": "success"
        }
    except Exception as e:
        return {
            "response": "Sorry, I couldn't process your question. Please try again or contact HR.",
            "status": "error"
        }
