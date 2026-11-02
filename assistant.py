"""FAQ chat graph for the Nirbhor Bank assistant."""
import csv
import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

FAQ_PATH = Path(os.getenv("FAQ_PATH", Path(__file__).parent / "bank-faq.csv"))
GROQ_MODEL = os.getenv("MODEL_NAME")
EMBED_MODEL = os.getenv("EMBED_MODEL")
TOP_K = 4

ANSWER_PROMPT = """You are the customer support assistant of Nirbhor Bank PLC (mobile app: NB Go).
    Answer only from the FAQ entries below. Do not guess or invent bank policies.
    Reply in the same language and script as the latest customer message:
    - English in English
    - Bangla script in Bangla script
    - Banglish in Latin script, preserving the customer's Banglish style
    If the FAQ does not answer the question, politely say so in the customer's language and offer to connect them to a human customer service officer. This includes other people's banking information, unsupported products or services, live prices, personal exceptions, and anything else not in the FAQ. If part is answered, answer that part and offer a person for the rest.
    Keep amounts, percentages, time limits, hotline 16XXX, and app name NB Go as written in the FAQ. Be brief and friendly.

    FAQ ENTRIES
{context}"""

PRIVACY_CHECK_PROMPT = """Classify whether the latest request asks for confidential account, balance, transaction, or personal banking information about another person (including a friend or family member). The customer asking about their own account is not a privacy request. Understand English, Bangla, and Banglish. Reply with exactly PRIVATE or OTHER."""

PRIVACY_REPLIES = {
    "english": "Sorry, I can’t share another person’s account or banking information because it’s confidential. Please ask them to contact us directly.",
    "banglish": "Dukhito, onno karo account ba banking information confidential, tai ami share korte parbo na. Taderke directly amader sathe contact korte bolun.",
    "bangla": "দুঃখিত, অন্য কারও অ্যাকাউন্ট বা ব্যাংকিং তথ্য গোপনীয়, তাই আমি তা জানাতে পারি না। তাঁকে সরাসরি আমাদের সঙ্গে যোগাযোগ করতে বলুন।",
}


def load_faq(path: str | Path = FAQ_PATH) -> list[Document]:
    with open(path, newline="", encoding="utf-8-sig") as file:
        return [
            Document(
                page_content=(
                    f"Topic: {row['topic']}\nQuestion: {row['question']}\n"
                    f"Answer: {row['answer']}"
                ),
                metadata={"id": int(row["id"]), "topic": row["topic"]},
            )
            for row in csv.DictReader(file)
        ]


def detect_reply_language(text: str) -> str:
    if any("\u0980" <= char <= "\u09ff" for char in text):
        return "bangla"
    text = f" {text.lower()} "
    markers = (
        " amar ", " amr ", " aamr ", " apnar ", " apnader ", " koto ",
        " kivabe ", " kibhabe ", " korbo ", " korte ", " chai ", " taka ",
        " er ", " bolun ", " janbo ", " hobe ", " ache ", " ki ", " bondhur ",
    )
    return "banglish" if any(marker in text for marker in markers) else "english"


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    sources: list[int]
    search_query: str


def load_resources():
    documents = load_faq()
    vector_store = InMemoryVectorStore(
        GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    )
    vector_store.add_documents(documents)
    return vector_store, len(documents)


vector_store, FAQ_COUNT = load_resources()
llm = ChatGroq(model=GROQ_MODEL, temperature=0)
checkpointer = InMemorySaver()


def chat_node(state: ChatState):
    latest = next(message for message in reversed(state["messages"]) if message.type == "human")
    privacy_result = llm.invoke([
        SystemMessage(PRIVACY_CHECK_PROMPT),
        HumanMessage(str(latest.content)),
    ])

    if privacy_result.content.strip().upper().startswith("PRIVATE"):
        language = detect_reply_language(str(latest.content))
        return {
            "messages": [AIMessage(PRIVACY_REPLIES[language])],
            "sources": [],
            "search_query": "",
        }

    history = state["messages"][-10:]
    transcript = "\n".join(
        f"{'Customer' if message.type == 'human' else 'Assistant'}: {message.content}"
        for message in history
    )
    rewrite = llm.invoke([
        SystemMessage(
            "Rewrite the conversation as one standalone English FAQ search query. "
            "Translate Bangla and Banglish to English and use context for follow-ups. "
            "Return only the query."
        ),
        HumanMessage(transcript),
    ])
    documents = vector_store.similarity_search(rewrite.content.strip(), k=TOP_K)
    context = "\n\n".join(
        f"[FAQ #{doc.metadata['id']}]\n{doc.page_content}" for doc in documents
    )
    answer = llm.invoke([
        SystemMessage(ANSWER_PROMPT.format(context=context)),
        *history,
    ])
    return {
        "messages": [answer],
        "sources": [doc.metadata["id"] for doc in documents],
        "search_query": rewrite.content.strip(),
    }


graph_builder = StateGraph(ChatState)
graph_builder.add_node("chat_node", chat_node)
graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)
chatbot = graph_builder.compile(checkpointer=checkpointer)


def get_all_threads():
    return list({
        checkpoint.config["configurable"]["thread_id"]
        for checkpoint in checkpointer.list(None)
    })
