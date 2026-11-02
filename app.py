import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
from assistant import build_graph, build_vector_store, load_faq  # noqa: E402

FAQ_PATH = Path(os.getenv("FAQ_PATH", Path(__file__).parent / "bank-faq.csv"))

st.set_page_config(page_title="NB Go Assistant", page_icon="🏦")

for key in ("GROQ_API_KEY", "GOOGLE_API_KEY"):
    if not os.getenv(key):
        st.error(f"Missing {key}. Add it to your .env file.")
        st.stop()


def get_current_chat():
    """Read the active conversation from its LangGraph checkpoint."""
    snapshot = st.session_state.graph.get_state(
        {"configurable": {"thread_id": st.session_state.thread_id}}
    )
    if not snapshot or not snapshot.values:
        return []
    chat = []
    for message in snapshot.values.get("messages", []):
        if isinstance(message, HumanMessage):
            chat.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage):
            chat.append({"role": "assistant", "content": str(message.content)})
    return chat


# Streamlit reruns the script after each interaction. Store the graph and its
# in-memory checkpointer in this browser session instead of using cache_resource.
if "graph" not in st.session_state:
    docs = load_faq(FAQ_PATH)
    store = build_vector_store(docs)
    st.session_state.graph = build_graph(store, InMemorySaver())
    st.session_state.faq_count = len(docs)
if "thread_ids" not in st.session_state:
    st.session_state.thread_ids = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if st.session_state.thread_id not in st.session_state.thread_ids:
    st.session_state.thread_ids.append(st.session_state.thread_id)

graph = st.session_state.graph

with st.sidebar:
    st.header("🏦 Nirbhor Bank")
    st.caption(f"{st.session_state.faq_count} FAQ entries loaded from `{FAQ_PATH.name}`")
    if st.button("New chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.thread_ids.append(st.session_state.thread_id)
        st.rerun()

    st.caption("Recent chats")
    for thread_id in reversed(st.session_state.thread_ids):
        if st.button(thread_id[:20], key=f"thread-{thread_id}", use_container_width=True):
            st.session_state.thread_id = thread_id
            st.rerun()

st.title("NB Go Assistant")
st.caption("Ask in English, বাংলা, or Banglish.")

for message in get_current_chat():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question…"):
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
        if result.get("sources"):
            with st.expander("Retrieved FAQ entries"):
                st.write(", ".join(f"#{i}" for i in result["sources"]))
    st.rerun()
