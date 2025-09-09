import os
from dotenv import load_dotenv
from langchain_chroma  import Chroma
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
import pandas as pd
import streamlit as st

# ================================
# Load environment variables
# ================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
system_prompt_path = os.getenv("SYSTEM_PROMPT_PATH")
db_path = os.getenv("DB_PATH")
collection = os.getenv("COLLECTION_NAME")
FEEDBACK_FILE = os.getenv("DOCTOR_FEEDBACK_FILE_PATH")

if not system_prompt_path or not db_path or not collection:
    raise ValueError("❌ Please set SYSTEM_PROMPT_PATH, DB_PATH, and COLLECTION_NAME in .env")

# Normalize paths
system_prompt_path = os.path.normpath(system_prompt_path)
db_path = os.path.normpath(db_path)

# ================================
# Load system prompt
# ================================
with open(system_prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ]
)

# ================================
# Initialize LLM
# ================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.5,
    max_output_tokens=4096
)

output_parser = StrOutputParser()

# ================================
# Functions
# ================================
def load_db(embeddings):
    """Load Chroma DB based on user choice."""

    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection
    )
    return db


def build_chain(db):
    """Build retrieval + LLM chain."""
    retriever = db.as_retriever()

    retrieval_chain = RunnableMap({
        "question": lambda x: x["question"],
        "context": lambda x: "\n\n".join(
            doc.page_content for doc in retriever.invoke(x["question"])
        ),
        "sources": lambda x: [
            doc.metadata.get("source", "unknown") for doc in retriever.invoke(x["question"])
        ],
        "history": lambda x: x.get("history", [])
    })

    chain = retrieval_chain | prompt | llm | output_parser
    return chain, retriever


# ================================
# Conversational History Functions
# ================================


# Store histories for multiple sessions
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create a session-based chat history."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def build_conversational_chain(db, session_id: str = "default"):
    chain, retriever = build_chain(db)
    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",   # 👈 tell it which key has the chat input
        history_messages_key="history",  # 👈 optional, for saving/retrieving
    )
    config = {"configurable": {"session_id": session_id}}
    return with_history, retriever, config


# ================================
# Feedback Logging Functions
# ================================


def save_feedback(query: str, bot_output: str, doctor_feedback: str):
    """Save feedback into an Excel file."""
    data = {
        "query": [query],
        "chatbot_output_summary": [bot_output],
        "doctor_feedback": [doctor_feedback],
    }

    # If file exists, append; else create
    try:
        df_existing = pd.read_excel(FEEDBACK_FILE)
        df_new = pd.DataFrame(data)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_combined = pd.DataFrame(data)

    df_combined.to_excel(FEEDBACK_FILE, index=False)

