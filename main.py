import os, re, time
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain.schema import messages_to_dict, messages_from_dict

# ❶ ENV + INITIALIZATION
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    convert_system_message_to_human=True
)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)

# ❷ MODELS
class UserProfile(BaseModel):
    name: str = "Friend"
    language: str = "English"
    interests: List[str] = []

class Query(BaseModel):
    question: str

# ❸ MEMORY HELPERS

def serialize_messages(msgs: List[BaseMessage]) -> List[Dict[str, Any]]:
    return messages_to_dict(msgs)

def deserialize_messages(data: List[Dict[str, Any]]) -> List[BaseMessage]:
    return messages_from_dict(data)

def get_memory(uid: str, llm_instance: ChatGoogleGenerativeAI) -> ConversationSummaryBufferMemory:
    mem = ConversationSummaryBufferMemory(
        llm=llm_instance,
        max_token_limit=350,
        memory_key="chat_history",
        return_messages=True
    )
    return mem

def save_memory(uid: str, mem: ConversationSummaryBufferMemory):
    pass  # Firebase removed, nothing to save

# ❹ PROFILE HELPERS
PROFILE_DB: Dict[str, UserProfile] = {}  # Local memory storage

def get_profile(uid: str) -> UserProfile:
    return PROFILE_DB.get(uid, UserProfile())

def save_profile(uid: str, profile: UserProfile):
    PROFILE_DB[uid] = profile

_NAME_RE = re.compile(r"\b(?:i am|i'm|my name is|call me|this is)\s+([A-Za-z]+)", re.I)
_LANG_RE = re.compile(r"\b(?:i speak|my language is|in)\s+([A-Za-z]+)\b", re.I)
_INT_RE = re.compile(r"\b(?:i (?:like|love|enjoy)|i'm interested in)\s+([^\.]+)", re.I)

def extract_profile(msgs: List[BaseMessage], profile: UserProfile) -> UserProfile:
    updated = profile.copy()
    texts = [m.content.lower() for m in msgs if hasattr(m, "content")]
    for t in texts:
        if updated.name == "Friend" and (m := _NAME_RE.search(t)):
            updated.name = m.group(1).title()
        if updated.language == "English" and (m := _LANG_RE.search(t)):
            updated.language = m.group(1).title()
        for m in _INT_RE.findall(t):
            for intr in [i.strip().title() for i in re.split(r",|and", m) if i.strip()]:
                if intr and intr not in updated.interests:
                    updated.interests.append(intr)
    return updated

# ❺ PROMPT

def make_prompt(profile: UserProfile) -> PromptTemplate:
    tmpl = f"""
You are DotsBit Assistant — an intelligent, fluent, and highly professional virtual assistant.
Your tone is warm, articulate, and confident. Help the user clearly and truthfully,
based on the information provided.

You are speaking with {profile.name} who prefers {profile.language}.
{f"Known interests: {', '.join(profile.interests)}." if profile.interests else ""}

Please do NOT mention the words "context", "documents", or "sources".
If you have no answer, say:
"I'm sorry, I don't have information about that at the moment."

Do not guess or make up facts. Keep answers direct, helpful, and human‑like.

---
Context:
{{context}}

Question:
{{question}}

Answer:
"""
    return PromptTemplate.from_template(tmpl)

# ❻ QUICK REPLIES
GREET = {"hi", "hello", "hey", "salam", "assalamualaikum", "hi dotsbit"}
FAREWELL = {"bye", "goodbye", "see you", "talk to you later"}
THANKS = {"thanks", "thank you", "shukriya", "jazakallah", "thank you so much"}

def quick_reply(txt: str, p: UserProfile) -> str | None:
    txt = txt.strip().lower()
    if txt in GREET:
        return f"Hello{', ' + p.name if p.name != 'Friend' else ''}! Welcome to DotsBit. How can I assist you today?"
    if txt in THANKS:
        return f"You're welcome{', ' + p.name if p.name != 'Friend' else ''}! If you have any other questions, feel free to ask."
    if txt in FAREWELL:
        return f"Thank you for chatting with us{', ' + p.name if p.name != 'Friend' else ''}. Have a great day!"
    return None

# ❼ FASTAPI APP
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.post("/chat/{user_id}")
async def chat(user_id: str, query: Query):
    if not query.question.strip():
        raise HTTPException(400, "No question provided")

    print("⚠️ Skipping token verification for testing/demo mode")

    memory = get_memory(user_id, llm)
    profile = get_profile(user_id)

    updated = extract_profile(memory.chat_memory.messages, profile)
    if updated != profile:
        save_profile(user_id, updated)
        profile = updated

    if (resp := quick_reply(query.question, profile)):
        return {"answer": resp}

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": make_prompt(profile)},
        verbose=False,
    )

    try:
        answer = qa_chain.run(query.question.strip())
        if not answer:
            answer = "I'm sorry, I don't have information about that at the moment."
    except Exception as e:
        print("LLM error:", e)
        answer = "I'm sorry, something went wrong while generating the answer."

    save_memory(user_id, memory)
    return {"answer": answer}

@app.get("/")
async def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
