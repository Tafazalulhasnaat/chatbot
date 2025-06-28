import os, re, time, inspect
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai

import firebase_admin
from firebase_admin import credentials, firestore, auth

from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
# --- MODIFIED: Import ConversationSummaryBufferMemory instead of ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain.schema import messages_to_dict, messages_from_dict

# ❶ ENV + INITIALISATION

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    convert_system_message_to_human=True
)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)

cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

TOKEN_LEEWAY_SECONDS = 10  # ⏳ Grace period for token expiry


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

# --- MODIFIED: The get_memory function is updated to use ConversationSummaryBufferMemory
def get_memory(uid: str, llm_instance: ChatGoogleGenerativeAI) -> ConversationSummaryBufferMemory:
    """
    Initializes a summarization buffer memory for the user.
    It keeps a buffer of recent messages and summarizes older ones.
    """
    # This memory type needs the LLM to create summaries.
    # max_token_limit controls the size of the recent message buffer.
    mem = ConversationSummaryBufferMemory(
        llm=llm_instance,
        max_token_limit=500,
        memory_key="chat_history",
        return_messages=True
    )
    doc = db.collection("user_memories").document(uid).get()
    if doc.exists and (hist := doc.to_dict().get("chat_history")):
        try:
            mem.chat_memory.messages = deserialize_messages(hist)
        except Exception as e:
            print("Memory deserialisation failed:", e)
    return mem

# --- MODIFIED: Updated the type hint for clarity. The function body is unchanged.
def save_memory(uid: str, mem: ConversationSummaryBufferMemory):
    db.collection("user_memories").document(uid).set(
        {"chat_history": serialize_messages(mem.chat_memory.messages)}
    )


# ❹ PROFILE HELPERS (No changes in this section)

PROFILE_COLL = db.collection("user_profiles")

def get_profile(uid: str) -> UserProfile:
    doc = PROFILE_COLL.document(uid).get()
    return UserProfile(**doc.to_dict()) if doc.exists else UserProfile()

def save_profile(uid: str, profile: UserProfile):
    PROFILE_COLL.document(uid).set(profile.dict())

# Very naïve regex extractor – swap with NLP / LLM for production
_NAME_RE   = re.compile(r"\b(?:i am|i'm|my name is|call me|this is)\s+([A-Za-z]+)", re.I)
_LANG_RE   = re.compile(r"\b(?:i speak|my language is|in)\s+([A-Za-z]+)\b", re.I)
_INT_RE    = re.compile(r"\b(?:i (?:like|love|enjoy)|i'm interested in)\s+([^\.]+)", re.I)

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


# ❺ PROMPT (No changes in this section)

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


# ❻ MISC CHAT UTIL (No changes in this section)
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


# ❼ TOKEN VERIFICATION with 10 s grace (No changes in this section)

def verify_id_token_with_grace(token: str) -> Dict[str, Any]:
    """
    Try firebase_admin.auth.verify_id_token with a 10‑second grace period.
    If the SDK supports 'clock_skew' (added mid‑2024), use it; otherwise
    fall back to manual check on ExpiredIdTokenError.
    """
    sig = inspect.signature(auth.verify_id_token)
    if "clock_skew" in sig.parameters:                      # new SDK
        return auth.verify_id_token(token, clock_skew=TOKEN_LEEWAY_SECONDS)

    try:                                                    # older SDK
        return auth.verify_id_token(token)
    except auth.ExpiredIdTokenError as e:
        # Token officially expired – allow if within grace
        # Decode without verification to read 'exp'
        try:
            import google.auth.jwt as google_jwt
            claims = google_jwt.decode(token, verify=False)
            if time.time() - claims.get("exp", 0) <= TOKEN_LEEWAY_SECONDS:
                return claims
        except Exception:
            pass
        raise e


# ❽ FASTAPI APP

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.post("/chat/{user_id}")
async def chat(user_id: str, query: Query, authorization: str = Header(None)):
    # 1. Basic validation
    if not query.question.strip():
        raise HTTPException(400, "No question provided")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")

    # 2. Verify token (with grace)
    token = authorization.split("Bearer ")[1]
    try:
        claims = verify_id_token_with_grace(token)
        if claims["uid"] != user_id:
            raise HTTPException(403, "User ID does not match token")
    except Exception as e:
        raise HTTPException(401, f"Invalid token: {e}")

    # 3. Load memory & profile
    # --- MODIFIED: Pass the global llm object to get_memory
    memory = get_memory(user_id, llm)
    profile = get_profile(user_id)

    # 4. Update profile from new history
    updated = extract_profile(memory.chat_memory.messages, profile)
    if updated != profile:
        save_profile(user_id, updated)
        profile = updated

    # 5. Quick greetings / farewells
    if (resp := quick_reply(query.question, profile)):
        return {"answer": resp}

    # 6. Build RAG chain
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

    # 7. Persist memory & return
    save_memory(user_id, memory)
    return {"answer": answer}

@app.get("/")
async def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)