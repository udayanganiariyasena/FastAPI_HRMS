from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
import requests

app = FastAPI()

class Question(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"message": "AIHRMS for HORIZON CAMPUS.."}

@app.post("/")
async def ask_hr(question: Question):
    answer = process_question(question.question)
    return {"response": answer}

def process_question(query):
    return f"Processed response for: {query}"
