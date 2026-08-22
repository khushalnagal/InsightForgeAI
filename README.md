# InsightForgeAI

Customer intelligence and risk analytics system for a financial services
business. Ingests real CFPB complaint data alongside transaction history,
engineers behavioral features, segments customers, scores churn risk with
an explainable model, and turns those explanations into plain-English
guidance without exposing the raw score to the reader.

---

## Architecture

    Raw Data (CFPB complaints + Kaggle transactions)
            |
    Feature Engineering (7 behavioral signals)
            |
    Segmentation (KMeans + PCA)  -->  4 customer segments, stored in MySQL
            |
    Churn Model (XGBoost + SMOTE, AUC 0.95)  -->  per-customer risk score
            |
    SHAP Explainability  -->  top 3 risk drivers per customer
            |
    GenAI Reason Cards (Groq/Llama)  -->  plain-English explanation, no raw score
            |
    API Server (FastAPI) + Query Agent (LangGraph)  -->  ask questions in natural language

---

## Customer Segments

KMeans, k=4 (silhouette 0.3455). Clusters are ranked and named by a
composite of average spend and spend trend, so the labels reflect
commercial standing, not churn risk directly (see note below).

| Segment | Count | Avg churn probability |
|---|---|---|
| Declining Disengaged | 4,014 | 24.0% |
| High Value At Risk | 3,137 | 23.3% |
| Premium Growth | 1,790 | 23.5% |
| Stable Mid Tier | 1,059 | 23.0% |

**Segmentation and churn are two independent models, not a pipeline.**
Both are trained on the same underlying behavioral features
(spend level, trend, volatility, recency), but the churn model doesn't
take `segment_label` as an input so segment membership and churn
probability only loosely correlate here and that's expected: clustering
optimizes for compact behavioral groups, while XGBoost optimizes
specifically for churn separability. Chaining them (e.g. training churn
per-segment, or feeding the label into churn as a feature) is the natural
next iteration if the two were meant to reinforce each other.

## Features Engineered

| Feature | What it captures |
|---|---|
| avg_monthly_spend | Baseline spending level |
| spend_trend | Growth or decline over the last 6 months (linear slope) |
| spend_volatility | How erratic the spending pattern is |
| top_category | Highest-spend merchant category |
| days_since_last_txn | Recency signal |
| txn_count_90d | Recent engagement |
| high_value_txn_count | Spending intensity |

## Churn Model

XGBoost + SMOTE (to correct class imbalance) → AUC 0.95. Churn probability
is written back to MySQL per customer. Top drivers: spend trend decline,
days since last transaction, low average monthly spend.

## Explainability

SHAP `TreeExplainer` produces a global feature-importance summary and a
per-customer waterfall of the top 3 drivers for every flagged customer.
Those 3 drivers - not the raw probability are what get passed to the
reason-card step.

## GenAI Reason Cards

The top-3 SHAP drivers for each at-risk customer go to an LLM (Groq /
Llama 3.3) with instructions to produce a 3-sentence explanation: why the
customer is at risk, what pattern their behavior shows, and what action to
take in plain business language, with no model score or jargon exposed.

## API Server

A FastAPI service exposes read-only, parameterised endpoints over the
results (segment summary, top churn risks, churn rate by segment,
complaint stats, and reason-card lookup by customer ID). There's no
free-text SQL endpoint every query is fixed, so the API can't be used to
run arbitrary database queries.

## Query Agent

A small LangGraph agent lets you ask questions in plain English
("who are the highest-risk customers?") instead of hitting the REST
endpoints directly. Three nodes: classify the question against a fixed
list of known intents → run the matching query → summarize the result in
plain English. Same "explain, don't dump the number" pattern as the
reason cards.

---

## Setup

    git clone https://github.com/khushalnagal/InsightForgeAI.git
    cd InsightForgeAI
    conda create -n customer-pulse python=3.11
    conda activate customer-pulse
    pip install -r requirements.txt

    cp .env.example .env
    # Fill in: MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB, GROQ_API_KEY

    mysql -u root -p < database/schema.sql

    python ingestion/kaggle_loader.py
    python ingestion/cfpb_api.py
    python processing/clean_complaints.py
    python features/engineering.py
    python models/segmentation.py
    python models/churn_model.py
    python explainability/shap_analysis.py
    python explainability/reason_cards.py

    # then, to explore results:
    uvicorn agents.api_server:app --reload --port 8000
    python agents/query_agent.py

---

## Tech Stack

| Layer | Technology |
|---|---|
| Storage | MySQL |
| Feature Engineering | Pandas, NumPy, SQLAlchemy |
| Segmentation | Scikit-learn (KMeans, PCA) |
| Churn Model | XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Reason Cards | Groq API (gpt-oss-20b) |
| API Server | FastAPI, Uvicorn |
| Query Agent | LangGraph |

---

## Numbers

10,000 customers · 284,807 transactions · ~1,000 CFPB complaints · AUC 0.95 · 4 segments

---

## Author

Khushal Nagal — [GitHub](https://github.com/khushalnagal) | [LinkedIn](https://linkedin.com/in/khushalnagal)
