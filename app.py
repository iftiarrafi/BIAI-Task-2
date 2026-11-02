import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from assistant import FAQ_COUNT, chatbot, get_all_threads

st.set_page_config(page_title="NB Go Assistant", page_icon="🏦")


def load_conversation(thread_id: str):
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    if not state or not state.values:
        return []

    messages = []
    for message in state.values.get("messages", []):
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        messages.append({"role": role, "content": str(message.content)})
    return messages


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "new_threads" not in st.session_state:
    st.session_state.new_threads = [st.session_state.thread_id]

with st.sidebar:
    st.header("🏦 Nirbhor Bank")
    st.caption(f"{FAQ_COUNT} FAQ entries loaded")

    if st.button("New chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.new_threads.append(st.session_state.thread_id)
        st.rerun()

    st.caption("Recent chats")
    thread_ids = list(dict.fromkeys(get_all_threads() + st.session_state.new_threads))
    for thread_id in reversed(thread_ids):
        if st.button(thread_id[:20], key=f"thread-{thread_id}", use_container_width=True):
            st.session_state.thread_id = thread_id
            st.rerun()

st.title("NB Go Assistant")
st.caption("Ask in English, বাংলা, or Banglish.")

for message in load_conversation(st.session_state.thread_id):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question…"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = chatbot.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
        st.markdown(result["messages"][-1].content)
        if result.get("sources"):
            with st.expander("Retrieved FAQ entries"):
                st.write(", ".join(f"#{source}" for source in result["sources"]))
    st.rerun()
