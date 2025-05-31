import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True)

# Load FAISS vector store with embeddings
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)

# Custom prompt for fluent and powerful assistant behavior
custom_prompt = PromptTemplate.from_template(
    """
    You are DotsBit Assistant — an intelligent, fluent, and highly professional virtual assistant.
    Your tone is warm, articulate, and confident. Your job is to help users by answering questions clearly,
    based on the information provided.

    Please do NOT mention the words "context", "documents", or "sources" in your response.
    If the answer is not known or missing, politely respond:
    "I'm sorry, I don’t have information about that at the moment."

    Do not guess or make up facts. Keep answers direct, helpful, and human-like.

    ---

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

# Setup ConversationalRetrievalChain with memory and custom prompt
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    verbose=False
)

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

# GREETING AND INTENT LAYER
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

    try:
        response = qa_chain.run(cleaned_query)
        return response if response else "I'm sorry, I don’t have information about that at the moment."
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I'm sorry, something went wrong while generating the answer."

@app.post("/chat")
async def chat(query: Query):
    if not query.question.strip():
        return {"error": "No question provided"}
    answer = get_chat_response(query.question)
    return {"answer": answer}

# Serve frontend (if present in /static folder)
@app.get("/")
async def root():
    return FileResponse("static/index.html")
