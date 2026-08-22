import os
import mysql.connector
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

logger.add("logs/clean_complaints.log", rotation="1 MB", retention="7 days", level="INFO")


def clean_and_load_complaints():
    # ── Connect ────────────────────────────────────────────────────────────────────
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB", "customer_pulse")
    )
    cursor = conn.cursor()
    logger.info("MySQL connected ✅")

    # ── Load raw complaints ────────────────────────────────────────────────────────
    logger.info("Loading raw complaints...")
    df = pd.read_sql("SELECT * FROM complaints", conn)
    logger.info(f"Loaded {len(df)} raw records")

    # ── Clean ──────────────────────────────────────────────────────────────────────
    # Drop rows missing critical fields
    df.dropna(subset=["id", "date_received", "product", "company"], inplace=True)
    logger.info(f"After dropping nulls: {len(df)} records")

    # Standardize text columns safely
    for col in ["product", "sub_product", "issue", "company", "state", "company_response"]:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip().str.title()

    # Standardize state to uppercase safely
    df["state"] = df["state"].fillna("").str.upper()

    # Fix consumer_disputed safely
    df["consumer_disputed"] = df["consumer_disputed"].fillna("N/A").str.strip()

    # Add simple feature flags
    df["has_narrative"] = df["narrative"].notna() & (df["narrative"].fillna("").str.strip() != "")
    df["is_disputed"] = df["consumer_disputed"].str.upper() == "YES"
    df["is_resolved"] = df["company_response"].str.lower().str.contains("closed", na=False)

    df["has_narrative"] = df["has_narrative"].astype(int)
    df["is_disputed"] = df["is_disputed"].astype(int)
    df["is_resolved"] = df["is_resolved"].astype(int)

    # Ensure date_received is proper date
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce").dt.date
    df.dropna(subset=["date_received"], inplace=True)

    logger.info(f"Final clean record count: {len(df)}")

    # ── Create cleaned_complaints table ───────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cleaned_complaints (
            id                  VARCHAR(64) PRIMARY KEY,
            date_received       DATE,
            product             VARCHAR(128),
            sub_product         VARCHAR(128),
            issue               VARCHAR(256),
            narrative           TEXT,
            company             VARCHAR(256),
            state               VARCHAR(100),
            company_response    VARCHAR(256),
            consumer_disputed   VARCHAR(16),
            has_narrative       TINYINT DEFAULT 0,
            is_disputed         TINYINT DEFAULT 0,
            is_resolved         TINYINT DEFAULT 0,
            cleaned_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    logger.info("cleaned_complaints table ready ✅")

    # ── Upsert into cleaned_complaints ────────────────────────────────────────────
    query = """
        INSERT INTO cleaned_complaints
            (id, date_received, product, sub_product, issue, narrative,
             company, state, company_response, consumer_disputed,
             has_narrative, is_disputed, is_resolved)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            date_received     = VALUES(date_received),
            product           = VALUES(product),
            sub_product       = VALUES(sub_product),
            issue             = VALUES(issue),
            narrative         = VALUES(narrative),
            company           = VALUES(company),
            state             = VALUES(state),
            company_response  = VALUES(company_response),
            consumer_disputed = VALUES(consumer_disputed),
            has_narrative     = VALUES(has_narrative),
            is_disputed       = VALUES(is_disputed),
            is_resolved       = VALUES(is_resolved)
    """

    df = df.where(pd.notnull(df), None)

    rows = [
        (
            None if pd.isna(row.id) else row.id,
            None if pd.isna(row.date_received) else row.date_received,
            None if pd.isna(row.product) else row.product,
            None if pd.isna(row.sub_product) else row.sub_product,
            None if pd.isna(row.issue) else row.issue,
            None if pd.isna(row.narrative) else row.narrative,
            None if pd.isna(row.company) else row.company,
            None if pd.isna(row.state) else row.state,
            None if pd.isna(row.company_response) else row.company_response,
            None if pd.isna(row.consumer_disputed) else row.consumer_disputed,
            int(row.has_narrative),
            int(row.is_disputed),
            int(row.is_resolved)
        )
        for row in df.itertuples()
    ]

    cursor.executemany(query, rows)
    conn.commit()

    logger.info(f"Inserted/updated {cursor.rowcount} records into cleaned_complaints ✅")
    print(f"\n✅ Done! {len(df)} clean records saved to cleaned_complaints table")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    clean_and_load_complaints()
