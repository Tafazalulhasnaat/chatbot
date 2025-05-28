# main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")  # Free & fast Gemini Flash model

# Load FAISS vector store with embeddings
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend access (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify frontend URL like ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for incoming queries
class Query(BaseModel):
    question: str

def get_context(query: str, k: int = 3) -> str:
    """
    Retrieve top-k similar documents from vectorstore as context.
    """
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

def get_answer(query: str) -> str:
    """
    Generate answer from Gemini 1.5 Flash model using retrieved context.
    """
    context = get_context(query)
    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question:
{query}
"""
    response = model.generate_content(prompt)
    return response.text

@app.post("/chat")
async def chat(query: Query):
    """
    Endpoint to receive a question and return the generated answer.
    """
    if not query.question.strip():
        return {"error": "No question provided"}
    
    answer = get_answer(query.question)
    return {"answer": answer}

# Optional: serve frontend HTML if placed in 'static' folder
@app.get("/")
async def root():
    return FileResponse("static/index.html")
