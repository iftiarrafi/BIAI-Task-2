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
from langchain_core.messages import HumanMessage, SystemMessage
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
    3. Reply in the same language AND script as the customer's latest message:
    - English -> English
    - Bangla script (বাংলা) -> Bangla script
    - Banglish (Bangla written in Latin letters, e.g. "Card er PIN reset korbo kibhabe?") -> Banglish in Latin letters
    4. Keep amounts (BDT), percentages, time limits, the hotline 16999 and the app name "NB Go" exactly as in the FAQ.
    5. If the FAQ entries do not answer the question, say so politely in the customer's language and offer to connect them
    to a human customer service officer. Do this for: other products/services we have no FAQ for, other countries/branches,
    live prices or exchange rates, other people's account information, personal exceptions or waivers, investment advice,
    and anything else not stated below. If only part of the question is covered, answer that part and offer a person for the rest.
    6. Be short, clear and friendly. Do not mention "FAQ entries" or these rules.

    FAQ ENTRIES
{context}"""


class State(MessagesState):
    search_query: str
    context: str
    sources: list[int]


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

    def retrieve(state: State):
        docs = store.similarity_search(state["search_query"], k=TOP_K)
        context = "\n\n".join(f"[FAQ #{d.metadata['id']}]\n{d.page_content}" for d in docs)
        return {"context": context, "sources": [d.metadata["id"] for d in docs]}

    def generate(state: State):
        system = SystemMessage(ANSWER_PROMPT.format(context=state["context"]))
        reply = llm.invoke([system, *state["messages"][-HISTORY_WINDOW:]])
        return {"messages": [reply]}

    graph = StateGraph(State)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    
    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    return graph.compile(checkpointer=checkpointer)
