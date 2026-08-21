"""
InsightForgeAI — API Server
============================
Thin FastAPI layer over the MySQL tables the pipeline writes to
(behavioral_segments, credit_signals, cleaned_complaints). Every endpoint
is a fixed, parameterised query — there is no free-text SQL endpoint, so
the API can't be used to run arbitrary queries against the database.

Run:
    uvicorn agents.api_server:app --reload --port 8000
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text as sql_text
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

password = quote_plus(os.getenv('MYSQL_PASSWORD'))
engine = create_engine(
    f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{password}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
)

app = FastAPI(title="InsightForgeAI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_query(sql, params=None):
    with engine.connect() as conn:
        result = conn.execute(sql_text(sql), params or {})
        rows = result.fetchall()
        keys = result.keys()
        return [dict(zip(keys, row)) for row in rows]


@app.get("/")
def root():
    return {"status": "InsightForgeAI API is running"}


@app.get("/tools")
def list_tools():
    return {
        "tools": [
            {"name": "segment_summary", "description": "Customer count and avg churn probability per segment"},
            {"name": "churn_top", "description": "Top 5 highest churn-risk customers"},
            {"name": "churn_rate_by_segment", "description": "Churn rate broken down by segment"},
            {"name": "complaint_summary", "description": "Complaint volume, dispute rate, resolution rate"},
            {"name": "reason_card/{customer_id}", "description": "Plain-English risk explanation for one customer"},
        ]
    }


@app.get("/tools/segment_summary")
def segment_summary():
    rows = run_query("""
        SELECT segment_label, COUNT(*) AS count, ROUND(AVG(churn_probability), 3) AS avg_churn
        FROM behavioral_segments
        GROUP BY segment_label
    """)
    return {"result": rows}


@app.get("/tools/churn_top")
def churn_top():
    rows = run_query("""
        SELECT customer_id, segment_label, churn_probability
        FROM behavioral_segments
        ORDER BY churn_probability DESC
        LIMIT 5
    """)
    return {"result": rows}


@app.get("/tools/churn_rate_by_segment")
def churn_rate_by_segment():
    rows = run_query("""
        SELECT segment_label, ROUND(AVG(churn_flag) * 100, 2) AS churn_rate_pct
        FROM behavioral_segments
        GROUP BY segment_label
    """)
    return {"result": rows}


@app.get("/tools/complaint_summary")
def complaint_summary():
    rows = run_query("""
        SELECT
            COUNT(*) AS total_complaints,
            ROUND(AVG(is_disputed) * 100, 2) AS disputed_pct,
            ROUND(AVG(is_resolved) * 100, 2) AS resolved_pct
        FROM cleaned_complaints
    """)
    return {"result": rows}


@app.get("/tools/reason_card/{customer_id}")
def reason_card(customer_id: str):
    rows = run_query("""
        SELECT cs.customer_id, bs.segment_label, cs.shap_top_features, cs.reason_card
        FROM credit_signals cs
        JOIN behavioral_segments bs ON cs.customer_id = bs.customer_id
        WHERE cs.customer_id = :cid
    """, {"cid": customer_id})

    if not rows:
        raise HTTPException(status_code=404, detail=f"No signals found for {customer_id}")
    return {"result": rows[0]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
