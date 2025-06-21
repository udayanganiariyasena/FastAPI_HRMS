# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

app = FastAPI()

# Initialize RAG system
def initialize_rag():
    loader = PyPDFLoader("hr_document.pdf")
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    
    return RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )

rag_qa = initialize_rag()

@app.post("/ask")
async def ask_question(question: BaseModel):
    result = rag_qa({"query": question.question})
    return {"response": result["result"]}
