from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

# Predefined Q&A pairs
qa_pairs = {
    "hello": "Hello! How can I assist you with HRMS today?",
    "leave": "You can apply for leave through the Leave Management system.",
    "attendance": "Attendance can be marked through our biometric system or mobile app.",
    "salary": "Salary details are available in the Payroll section.",
    "benefits": "Employee benefits include health insurance, paid leave, and retirement plans.",
    "contact hr": "You can reach HR at hr@horizoncampus.com or extension 123.",
    "policies": "Company policies are available in the Documents section of your portal.",
    "holidays": "The holiday calendar is published in the Announcements section.",
}

@app.get("/")
def read_root():
    return {"message": "AIHRMS for HORIZON CAMPUS"}

@app.post("/ask")
async def ask_hr(question: Question):
    answer = process_question(question.question.lower())
    return {"response": answer}

def process_question(query: str):
    # Check for direct matches
    for question, answer in qa_pairs.items():
        if question in query:
            return answer
    
    # Add more sophisticated processing if needed
    if "leave" in query:
        return qa_pairs["leave"]
    elif "salary" in query:
        return qa_pairs["salary"]
    
    # Default response if no match found
    return "I couldn't find information about that. Please contact HR directly for specific queries."
