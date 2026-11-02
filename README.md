# NB Go Bank FAQ Assistant

A bilingual customer support chatbot prototype for **Nirbhor Bank PLC**, a fictional bank. Customers can ask banking FAQ questions in English, Bangla script, or Banglish and continue separate conversations from the sidebar.

## What it does

- Answers banking questions using a 30-entry FAQ dataset in `bank-faq.csv`.
- Uses semantic search to retrieve relevant FAQ entries before generating an answer.
- Prompts the assistant to reply in the language and script used by the customer.
- Checks requests for another person's private banking information and responds that the information is confidential.
- Falls back to offering human support when the FAQ does not cover a question.
- Provides a Streamlit chat interface with a welcome screen, conversation titles, and thread switching.

This is a demonstration project. It has no connection to real customer accounts, balances, transactions, or bank systems. Nirbhor Bank PLC and NB Go are fictional.

## How it works

1. `app.py` renders the Streamlit interface and sends each message with its conversation's thread ID.
2. `assistant.py` runs a LangGraph chat node. It first checks for a request about another person's private information.
3. For other requests, the model rewrites the conversation into an English search query. Google embeddings and an in-memory vector store retrieve the closest FAQ entries.
4. ChatGroq generates a short response using the retrieved FAQ context and the conversation history.
5. LangGraph's `InMemorySaver` keeps messages separated by thread while the app process is running.

The FAQ is the only source for bank-specific answers. Unsupported questions should receive a contact-support fallback instead of an invented bank policy.

## Technology

- Python and Streamlit
- LangGraph for the chat workflow and per-thread state
- LangChain `InMemoryVectorStore` and Google Generative AI embeddings for FAQ retrieval
- ChatGroq for privacy classification, query rewriting, and response generation
- CSV for the FAQ knowledge base

## Run locally

1. Create and activate a Python virtual environment.

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `env.example` to `.env` and fill in your API keys and model names:

   ```dotenv
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key
   MODEL_NAME=your_groq_chat_model
   EMBED_MODEL=your_google_embedding_model
   ```

4. Start the app from the project directory.

   ```bash
   streamlit run app.py
   ```

`FAQ_PATH` is optional and defaults to `bank-faq.csv` in the project directory.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and conversation selection |
| `assistant.py` | FAQ loading, retrieval, privacy check, prompts, and LangGraph chat node |
| `bank-faq.csv` | Bank information used to answer supported questions |
| `test-questions.csv` | Sample English, Bangla, and Banglish questions for manual evaluation |
| `requirements.txt` | Python dependencies |
| `env.example` | Environment variable template |

## Current limitations

- Conversation history is held in memory and is cleared when the app process restarts.
- The assistant does not authenticate users or access account-specific data; it cannot provide real balances or transaction details.
- The FAQ is in English, so answer quality depends on retrieval and the language-following behavior of the selected model.
