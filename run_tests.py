"""Run all questions in test-questions.csv through the graph and save results.csv for review."""
import csv
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
from assistant import build_graph, build_vector_store, load_faq  # noqa: E402

here = Path(__file__).parent
graph = build_graph(build_vector_store(load_faq(here / "bank-faq.csv")), InMemorySaver())

with open(here / "test-questions.csv", newline="", encoding="utf-8-sig") as f:
    questions = list(csv.DictReader(f))

rows = []
for q in questions:
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}  # fresh thread per question
    out = graph.invoke({"messages": [HumanMessage(q["question"])]}, config=cfg)
    rows.append(
        {
            "id": q["id"],
            "question": q["question"],
            "search_query": out["search_query"],
            "sources": ",".join(map(str, out["sources"])),
            "answer": out["messages"][-1].content,
        }
    )
    print(f"{q['id']:>2}. {q['question']}\n    -> {rows[-1]['answer']}\n")
    time.sleep(2)  # stay under free-tier rate limits

with open(here / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
