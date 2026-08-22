import os
import json
import time
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from urllib.parse import quote_plus
from groq import Groq
from loguru import logger

load_dotenv()

password = quote_plus(os.getenv('MYSQL_PASSWORD'))
engine = create_engine(
    f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{password}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
)

client = Groq(api_key=os.getenv('GROQ_API_KEY'))


def generate_reason_cards():
    """
    Reads SHAP top-3 features per at-risk customer (written by shap_analysis.py)
    and asks an LLM to turn them into a plain-English explanation — the model's
    churn probability is never surfaced to the reader, only the drivers and the
    recommended next step.
    """
    logger.info("Fetching at-risk customers without reason cards...")

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT cs.customer_id, cs.shap_top_features,
                   bs.churn_probability, bs.segment_label
            FROM credit_signals cs
            JOIN behavioral_segments bs ON cs.customer_id = bs.customer_id
            WHERE cs.reason_card IS NULL
            AND cs.shap_top_features IS NOT NULL
        """))
        rows = result.fetchall()

    logger.info(f"Found {len(rows)} customers needing reason cards...")

    generated = 0
    for row in rows:
        customer_id = row[0]
        shap_features = row[1]
        churn_prob = row[2]
        segment = row[3]

        try:
            shap_dict = json.loads(shap_features) if isinstance(shap_features, str) else shap_features

            prompt = f"""You are a senior risk analyst at a financial services company.
A customer has been flagged as high churn risk.

Customer segment: {segment}
Churn probability: {round(float(churn_prob) * 100, 1)}%
Top 3 risk signals (SHAP values): {json.dumps(shap_dict)}

Write a 3-sentence plain-English reason card explaining:
1. Why this customer is at risk based on the signals
2. What pattern their behavior shows
3. What action the retention team should take

Be specific, professional, and actionable. Do not mention the raw churn
probability or any model/SHAP terminology — write for a non-technical reader."""

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )

            reason_card = response.choices[0].message.content.strip()

            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE credit_signals
                    SET reason_card = :card
                    WHERE customer_id = :cid
                """), {"card": reason_card, "cid": customer_id})

            generated += 1

            if generated % 50 == 0:
                logger.info(f"Generated {generated}/{len(rows)} reason cards...")

            time.sleep(0.1)

        except Exception as e:
            if "tokens per day" in str(e) or "TPD" in str(e):
                logger.warning("Daily quota exhausted. Rerun tomorrow.")
                break
            logger.error(f"Failed for {customer_id}: {e}")
            time.sleep(2)
            continue

    logger.info(f"Done. Generated {generated} reason cards.")


if __name__ == "__main__":
    generate_reason_cards()
