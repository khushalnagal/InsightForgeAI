"""
InsightForgeAI — Query Agent
=============================
A small LangGraph agent that answers natural-language questions about the
customer base without exposing raw model internals to the caller.

Graph:
    classify_intent -> run_query -> summarize -> END

    1. classify_intent — Groq LLM maps the question to one of a fixed set
       of known intents (a closed list, not open-ended SQL generation).
    2. run_query        — the matched intent's parameterised SQL runs
       against MySQL.
    3. summarize         — Groq LLM turns the raw rows into a short,
       plain-English answer (same "explain the output, don't dump the
       score" pattern used by explainability/reason_cards.py).

Run interactively:
    python agents/query_agent.py
"""

import os
from typing import TypedDict, Optional
from sqlalchemy import create_engine, text as sql_text
from dotenv import load_dotenv
from urllib.parse import quote_plus
from groq import Groq
from loguru import logger
from langgraph.graph import StateGraph, END

load_dotenv()

password = quote_plus(os.getenv('MYSQL_PASSWORD'))
engine = create_engine(
    f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{password}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
)

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

INTENTS = """
segment_summary        - customer count and average churn probability per segment
churn_top               - the highest churn-risk customers
churn_rate_by_segment   - churn rate broken down by segment
complaint_summary       - complaint volume, dispute rate, resolution rate
unknown                 - the question doesn't match any of the above
"""

QUERY_MAP = {
    "segment_summary": """
        SELECT segment_label, COUNT(*) AS count, ROUND(AVG(churn_probability), 3) AS avg_churn
        FROM behavioral_segments GROUP BY segment_label
    """,
    "churn_top": """
        SELECT customer_id, segment_label, churn_probability
        FROM behavioral_segments ORDER BY churn_probability DESC LIMIT 5
    """,
    "churn_rate_by_segment": """
        SELECT segment_label, ROUND(AVG(churn_flag) * 100, 2) AS churn_rate_pct
        FROM behavioral_segments GROUP BY segment_label
    """,
    "complaint_summary": """
        SELECT COUNT(*) AS total_complaints,
               ROUND(AVG(is_disputed) * 100, 2) AS disputed_pct,
               ROUND(AVG(is_resolved) * 100, 2) AS resolved_pct
        FROM cleaned_complaints
    """,
}


class AgentState(TypedDict):
    question: str
    intent: Optional[str]
    rows: Optional[list]
    answer: Optional[str]


def classify_intent(state: AgentState) -> AgentState:
    prompt = f"""Match the user's question to exactly one intent name from this list:
{INTENTS}
Question: "{state['question']}"
Reply with only the intent name, nothing else."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip().lower()
    intent = next((k for k in QUERY_MAP if k in raw), "unknown")
    logger.info(f"Classified intent: {intent}")
    return {**state, "intent": intent}


def run_query(state: AgentState) -> AgentState:
    intent = state["intent"]
    if intent == "unknown":
        return {**state, "rows": []}

    with engine.connect() as conn:
        result = conn.execute(sql_text(QUERY_MAP[intent]))
        rows = [dict(zip(result.keys(), row)) for row in result.fetchall()]
    return {**state, "rows": rows}


def summarize(state: AgentState) -> AgentState:
    if state["intent"] == "unknown":
        return {**state, "answer": "I can only answer questions about segments, churn risk, and complaints right now."}

    if not state["rows"]:
        return {**state, "answer": "No data found for that question."}

    prompt = f"""A user asked: "{state['question']}"
Here is the query result: {state['rows']}
Answer their question in 1-2 plain-English sentences. Be direct and specific, no technical jargon."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.3,
    )
    return {**state, "answer": response.choices[0].message.content.strip()}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("run_query", run_query)
    graph.add_node("summarize", summarize)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "run_query")
    graph.add_edge("run_query", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


agent = build_graph()


def ask(question: str) -> str:
    result = agent.invoke({"question": question, "intent": None, "rows": None, "answer": None})
    return result["answer"]


if __name__ == "__main__":
    print("InsightForgeAI query agent — ask about segments, churn risk, or complaints. Ctrl+C to exit.")
    while True:
        try:
            q = input("\n> ")
            print(ask(q))
        except KeyboardInterrupt:
            break
