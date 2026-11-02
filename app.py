
"""Streamlit frontend for the Nirbhor Bank NB Go Assistant."""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from assistant import FAQ_COUNT, chatbot, get_all_threads


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NB Go Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    #MainMenu, footer, [data-testid="stHeader"] {
        visibility: hidden;
    }

    .stApp {
        background: var(--background-color);
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 900px;
        padding: 2rem 2rem 7rem;
    }

    /* SIDEBAR */

    [data-testid="stSidebar"] {
        background: var(--secondary-background-color);
        border-right: 1px solid rgba(128,128,128,.15);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.1rem .8rem;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 42px;
        text-align: left;
        justify-content: flex-start;
        border: 0;
        border-radius: 9px;
        background: transparent;
        padding: .55rem .8rem;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(128,128,128,.14);
    }

    [data-testid="stSidebar"] .stButton > button > div {
        width: 100%;
        justify-content: flex-start;
        text-align: left;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        text-align: left;
    }

    .brand {
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -.6px;
        padding: .4rem 0 1.5rem;
    }

    .brand-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 31px;
        height: 31px;
        margin-right: 8px;
        border-radius: 50%;
        background: #10a37f;
        color: #fff;
        font-size: 17px;
    }

    .sidebar-section {
        color: var(--text-color);
        opacity: .65;
        font-size: .78rem;
        font-weight: 600;
        margin: 1.3rem 0 .6rem .6rem;
    }

    /* WELCOME SCREEN */

    .welcome {
        padding-top: 15vh;
        text-align: center;
    }

    .welcome h1 {
        font-size: clamp(1.8rem,4vw,2.5rem);
        font-weight: 600;
        letter-spacing: -1px;
        margin-bottom: .6rem;
    }

    .welcome p {
        color: var(--text-color);
        opacity: .65;
        font-size: 1rem;
    }

    .welcome-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }

    .suggestion {
        padding: 14px 16px;
        border: 1px solid rgba(128,128,128,.23);
        border-radius: 14px;
        min-height: 85px;
        font-size: .88rem;
        color: var(--text-color);
        background: var(--secondary-background-color);
    }

    /* CHAT MESSAGES */

    [data-testid="stChatMessage"] {
        border: 0;
        padding: 1rem .25rem;
        background: transparent;
    }

    [data-testid="stChatMessageContent"] {
        line-height: 1.75;
        font-size: .96rem;
        overflow-wrap: anywhere;
    }

    [data-testid="stChatMessageContent"] pre {
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.2);
    }

    [data-testid="stChatMessageContent"] code {
        font-size: .88em;
    }

    /* CHAT INPUT */

    [data-testid="stChatInput"] {
        max-width: 850px;
        margin-left: auto;
        margin-right: auto;
    }

    [data-testid="stChatInput"] > div {
        border-radius: 24px;
        border: 1px solid rgba(128,128,128,.35);
        background: var(--secondary-background-color);
    }

    [data-testid="stChatInput"] textarea {
        padding-top: 14px;
        padding-bottom: 14px;
        font-size: .95rem;
    }

    /* FOOTER */

    .caption-text {
        opacity: .55;
        font-size: .76rem;
        text-align: center;
        margin-top: 1.1rem;
    }

    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 6rem;
        }

        .welcome {
            padding-top: 8vh;
        }

        [data-testid="stChatMessage"] {
            padding: .6rem 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONVERSATION MANAGEMENT
# ============================================================

def load_conversation(thread_id: str):

    state = chatbot.get_state(
        {"configurable": {"thread_id": thread_id}}
    )

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

        messages.append({
            "role": role,
            "content": str(message.content),
        })

    return messages


# ============================================================
# SESSION STATE
# ============================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "new_threads" not in st.session_state:
    st.session_state.new_threads = [
        st.session_state.thread_id
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class='brand'>
            <span class='brand-icon'>🏦</span>
            NB Go
        </div>
        """,
        unsafe_allow_html=True,
    )

    # New conversation

    if st.button("＋  New chat", use_container_width=True):

        st.session_state.thread_id = str(uuid.uuid4())

        st.session_state.new_threads.append(
            st.session_state.thread_id
        )

        st.rerun()

    st.markdown(
        "<div class='sidebar-section'>YOUR CHATS</div>",
        unsafe_allow_html=True,
    )

    # Retrieve conversation threads

    thread_ids = list(
        dict.fromkeys(
            get_all_threads() + st.session_state.new_threads
        )
    )

    # Display conversations

    for thread_id in reversed(thread_ids):

        conversation = load_conversation(thread_id)

        title = "New conversation"

        for message in conversation:

            if message["role"] == "user" and message["content"].strip():

                title = " ".join(
                    message["content"].split()
                )

                if len(title) > 32:
                    title = title[:29] + "..."

                break

        if thread_id == st.session_state.thread_id:
            title = "●  " + title

        if st.button(
            title,
            key=f"thread-{thread_id}",
            use_container_width=True,
        ):

            st.session_state.thread_id = thread_id
            st.rerun()

    st.divider()

    st.caption("Nirbhor Bank PLC")

    st.caption(
        f"Knowledge Base: {FAQ_COUNT} FAQ entries"
    )

    st.caption(
        "Supports English · বাংলা · Banglish"
    )


# ============================================================
# LOAD CURRENT CONVERSATION
# ============================================================

messages = load_conversation(
    st.session_state.thread_id
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not messages:

    st.markdown(
        """
        <div class='welcome'>
            <div class='welcome-icon'>🏦</div>
            <h1>How can I help?</h1>
            <p>
                Your banking assistant, ready to help.
                Ask me anything about NB Go.
            </p>
            <p>
                English · বাংলা · Banglish
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            <div class='suggestion'>
                <b>💳 Cards and payments</b><br><br>
                Ask about card fees, PINs, transfers,
                or payment issues.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            """
            <div class='suggestion'>
                <b>🏦 Accounts and services</b><br><br>
                Ask about account opening, balances,
                statements, or loans.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# DISPLAY CONVERSATION HISTORY
# ============================================================

else:

    for message in messages:

        with st.chat_message(
            message["role"],
            avatar=(
                "🧑‍💻"
                if message["role"] == "user"
                else "🏦"
            ),
        ):

            st.markdown(message["content"])


# ============================================================
# CHAT INPUT AND RESPONSE GENERATION
# ============================================================

if prompt := st.chat_input(
    "Message NB Go... (English, বাংলা, or Banglish)"
):

    # Display customer message

    with st.chat_message(
        "user",
        avatar="🧑‍💻",
    ):

        st.markdown(prompt)


    # Generate assistant response

    with st.chat_message(
        "assistant",
        avatar="🏦",
    ):

        with st.spinner("Thinking..."):

            try:

                result = chatbot.invoke(
                    {
                        "messages": [
                            HumanMessage(content=prompt)
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    },
                )

                response = result["messages"][-1].content

                st.markdown(response)

                # Display retrieved FAQ sources

                sources = result.get("sources", [])

                if sources:

                    with st.expander("Retrieved FAQ entries"):

                        st.write(
                            ", ".join(
                                f"#{source}"
                                for source in sources
                            )
                        )

            except Exception as error:

                st.error(
                    "Sorry, something went wrong. "
                    "Please try again."
                )

                # Log the actual error in the terminal
                print(f"Chatbot error: {error}")

    st.rerun()
