"""LangGraph pipeline for the Nirbhor Bank (NB Go) bilingual FAQ assistant.

Graph:  START -> rewrite_query -> retrieve -> generate -> END
Memory: InMemorySaver (per-thread conversation history)
RAG:    InMemoryVectorStore + Google embeddings, built from bank-faq.csv
LLM:    ChatGroq
"""
import csv
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

GROQ_MODEL = os.getenv("MODEL_NAME")
EMBED_MODEL = os.getenv("EMBED_MODEL")
TOP_K = 4
HISTORY_WINDOW = 10  # last N messages sent to the LLM

REWRITE_PROMPT = """You turn a bank customer's latest message into ONE standalone English search query for an FAQ database.
- The customer may write in English, Bangla script, or Banglish (Bangla in Latin letters). Translate to English.
- Use the earlier conversation to resolve follow-ups like "and for 12 months?" or "what about the fee?".
- Output ONLY the search query, nothing else."""

ANSWER_PROMPT = """You are the customer support assistant of Nirbhor Bank PLC (mobile app: NB Go).

    RULES
    1. Answer ONLY from the FAQ entries below. Never use outside knowledge, never guess, never invent numbers or policies.
    2. Do not stretch an FAQ rule to a different product or situation than the one asked about
    (e.g. the personal loan age limit does not tell you anything about a student loan).
    3. Reply in the same language, script, and writing style as the customer's latest message:
    - English -> English
    - Bangla script (বাংলা) -> Bangla script
    - Banglish (Bangla written in Latin letters, e.g. "Card er PIN reset korbo kibhabe?" or "Aamr current saving koto?") -> Banglish in Latin letters
    - Do not translate a Banglish reply into English or Bangla script. Apply this rule to fallback and contact-us replies too.
    4. Keep amounts (BDT), percentages, time limits, the hotline 16999 and the app name "NB Go" exactly as in the FAQ.
    5. If the FAQ entries do not answer the question, say so politely in the customer's language and offer to connect them
    to a human customer service officer. Do this for: other products/services we have no FAQ for, other countries/branches,
    live prices or exchange rates, other people's account information, personal exceptions or waivers, investment advice,
    and anything else not stated below. If only part of the question is covered, answer that part and offer a person for the rest.
    6. Be short, clear and friendly. Do not mention "FAQ entries" or these rules.

    FAQ ENTRIES
{context}"""

PRIVACY_CHECK_PROMPT = """Decide whether the customer's latest request seeks confidential account, balance, transaction, or other personal banking information about someone other than the customer. This includes a friend, family member, coworker, or any named/third party. Requests about the customer's own account are not third-party requests.
The customer may write in English, Bangla script, or Banglish. Understand all three.
Reply with exactly one token: PRIVATE or OTHER."""

PRIVACY_REPLIES = {
    "english": "Sorry, I can’t share another person’s account or banking information because it’s confidential. Please ask them to contact us directly.",
    "banglish": "Dukhito, onno karo account ba banking information confidential, tai ami share korte parbo na. Taderke directly amader sathe contact korte bolun.",
    "bangla": "দুঃখিত, অন্য কারও অ্যাকাউন্ট বা ব্যাংকিং তথ্য গোপনীয়, তাই আমি তা জানাতে পারি না। তাঁকে সরাসরি আমাদের সঙ্গে যোগাযোগ করতে বলুন।",
}


def _privacy_reply_language(text: str) -> str:
    """Choose the script/style for the fixed privacy response."""
    bangla_chars = sum("\u0980" <= char <= "\u09ff" for char in text)
    latin_letters = sum(char.isascii() and char.isalpha() for char in text)
    if bangla_chars:
        return "bangla"
    # Latin-script Bangla has common Bangla words. Only classify as Banglish
    # when there is evidence of Bangla rather than assuming every Latin query
    # is English.
    lowered = f" {text.lower()} "
    banglish_markers = (
        " amar ", " amr ", " aamr ", " apnar ", " apnader ", " koto ",
        " kivabe ", " kibhabe ", " korbo ", " korte ", " chai ", " taka ",
        " er ", " bolun ", " janbo ", " hobe ", " ache ", " ki ",
        " bondhur ", " bondhu ", " account e ", " hisab ",
    )
    return "banglish" if any(marker in lowered for marker in banglish_markers) else "english"


class State(MessagesState):
    search_query: str
    context: str
    sources: list[int]
    private_request: bool


def load_faq(path: str | Path) -> list[Document]:
    """Read the CSV fresh every time (utf-8-sig strips the BOM)."""
    docs = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            docs.append(
                Document(
                    page_content=f"Topic: {row['topic']}\nQuestion: {row['question']}\nAnswer: {row['answer']}",
                    metadata={"id": int(row["id"]), "topic": row["topic"]},
                )
            )
    return docs


def build_vector_store(docs: list[Document]) -> InMemoryVectorStore:
    store = InMemoryVectorStore(GoogleGenerativeAIEmbeddings(model=EMBED_MODEL))
    store.add_documents(docs)
    return store


def build_graph(store: InMemoryVectorStore, checkpointer: InMemorySaver):
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)

    def rewrite_query(state: State):
        recent = state["messages"][-HISTORY_WINDOW:]
        transcript = "\n".join(
            f"{'Customer' if m.type == 'human' else 'Assistant'}: {m.content}" for m in recent
        )
        out = llm.invoke([SystemMessage(REWRITE_PROMPT), HumanMessage(transcript)])
        return {"search_query": out.content.strip()}

    def check_privacy(state: State):
        latest = next(m for m in reversed(state["messages"]) if m.type == "human")
        out = llm.invoke([
            SystemMessage(PRIVACY_CHECK_PROMPT),
            HumanMessage(str(latest.content)),
        ])
        decision = out.content.strip().upper()
        return {"private_request": decision.startswith("PRIVATE")}

    def route_after_privacy_check(state: State):
        return "privacy_response" if state.get("private_request") else "rewrite_query"

    def privacy_response(state: State):
        latest = next(m for m in reversed(state["messages"]) if m.type == "human")
        language = _privacy_reply_language(str(latest.content))
        return {
            "messages": [AIMessage(PRIVACY_REPLIES[language])],
            "context": "",
            "sources": [],
        }

    def retrieve(state: State):
        docs = store.similarity_search(state["search_query"], k=TOP_K)
        context = "\n\n".join(f"[FAQ #{d.metadata['id']}]\n{d.page_content}" for d in docs)
        return {"context": context, "sources": [d.metadata["id"] for d in docs]}

    def generate(state: State):
        system = SystemMessage(ANSWER_PROMPT.format(context=state["context"]))
        reply = llm.invoke([system, *state["messages"][-HISTORY_WINDOW:]])
        return {"messages": [reply]}

    graph = StateGraph(State)
    graph.add_node("check_privacy", check_privacy)
    graph.add_node("privacy_response", privacy_response)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    
    graph.add_edge(START, "check_privacy")
    graph.add_conditional_edges("check_privacy", route_after_privacy_check)
    graph.add_edge("privacy_response", END)
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    return graph.compile(checkpointer=checkpointer)
