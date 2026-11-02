import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
from assistant import build_graph, build_vector_store, load_faq  # noqa: E402

FAQ_PATH = Path(os.getenv("FAQ_PATH", Path(__file__).parent / "bank-faq.csv"))

st.set_page_config(page_title="NB Go Assistant", page_icon="🏦")

for key in ("GROQ_API_KEY", "GOOGLE_API_KEY"):
    if not os.getenv(key):
        st.error(f"Missing {key}. Add it to your .env file.")
        st.stop()


@st.cache_resource
def get_checkpointer() -> InMemorySaver:
    return InMemorySaver()  # survives reruns, shared across threads


@st.cache_resource(show_spinner="Loading bank FAQ…")
def get_graph(faq_mtime: float):
    """Re-reads bank-faq.csv and rebuilds the vector store on app start,
    and again whenever the file's modified time changes."""
    docs = load_faq(FAQ_PATH)
    store = build_vector_store(docs)
    return build_graph(store, get_checkpointer()), len(docs)


graph, faq_count = get_graph(FAQ_PATH.stat().st_mtime)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []

with st.sidebar:
    st.header("🏦 Nirbhor Bank")
    st.caption(f"{faq_count} FAQ entries loaded from `{FAQ_PATH.name}`")
    if st.button("New chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat = []
        st.rerun()

st.title("NB Go Assistant")
st.caption("Ask in English, বাংলা, or Banglish.")

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Retrieved FAQ entries"):
                st.write(", ".join(f"#{i}" for i in m["sources"]))

if prompt := st.chat_input("Type your question…"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = graph.invoke(
                {"messages": [HumanMessage(prompt)]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
        reply = result["messages"][-1].content
        st.markdown(reply)
        
    st.session_state.chat.append(
        {"role": "assistant", "content": reply, "sources": result["sources"]}
    )
    st.rerun()
