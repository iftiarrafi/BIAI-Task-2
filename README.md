# Nirbhor Bank FAQ Assistant

"Nirbhor Bank PLC" and "NB Go" are fictional.

- `bank-faq.csv` contains the English bank FAQ and is the assistant's only source for banking answers.
- `test-questions.csv` contains sample English, Bangla, and Banglish questions.

The assistant replies in the customer's language and script. Requests for another person's confidential banking information receive a confidentiality response. Unsupported questions are answered with an offer to connect the customer to a person.

The Streamlit app uses an in-memory LangGraph checkpointer. Thread IDs appear in the sidebar, and conversations remain available while the app process is running. Restarting the process clears them.
