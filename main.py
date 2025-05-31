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
    allow_origins=["*"],
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
    Avoid mentioning context or saying "based on the text".
    """
    context = get_context(query)

    prompt = f"""
You are a helpful assistant for a company called DotsBit.

Use the context below to answer the user's question. 
If the answer is not found in the context, respond naturally with:
"I'm sorry, I don’t have information about that at the moment."

Do not mention the context or documents in your answer.

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()

        # Fallback if the model returns something empty or irrelevant
        if not answer or "based on the context" in answer.lower() or "not mentioned" in answer.lower():
            return "I'm sorry, I don’t have information about that at the moment."
        return answer

    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I'm sorry, something went wrong while generating the answer."

# GREETING LAYER
def get_chat_response(query: str) -> str:
    greetings = ["hi", "hello", "hey", "salam", "assalamualaikum", "hi dotsbit"]
    farewells = ["bye", "goodbye", "see you", "talk to you later"]
    thanks = ["thanks", "thank you", "shukriya", "jazakallah", "thank you so much"]

    cleaned_query = query.strip().lower()

    if cleaned_query in greetings:
        return "Hello! Welcome to DotsBit. How can I assist you today?"
    elif cleaned_query in thanks:
        return "You're welcome! If you have any other questions, feel free to ask."
    elif cleaned_query in farewells:
        return "Thank you for chatting with us. Have a great day!"
    return get_answer(query)

@app.post("/chat")
async def chat(query: Query):
    """
    Endpoint to receive a question and return the generated answer or greeting.
    """
    if not query.question.strip():
        return {"error": "No question provided"}
    
    answer = get_chat_response(query.question)
    return {"answer": answer}

# Optional: serve frontend HTML if placed in 'static' folder
@app.get("/")
async def root():
    return FileResponse("static/index.html")
