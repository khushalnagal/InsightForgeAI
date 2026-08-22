import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
from loguru import logger
import zipfile
import subprocess

load_dotenv()
logger.add("logs/kaggle_loader.log", rotation="1 MB", level="INFO")

# ── MySQL engine ──────────────────────────────────────────────────────────────
from urllib.parse import quote_plus

def get_engine():
    user = os.getenv("MYSQL_USER", "root")
    pw   = quote_plus(os.getenv("MYSQL_PASSWORD"))  # ✅ encodes @ and # safely
    host = os.getenv("MYSQL_HOST", "localhost")
    db   = os.getenv("MYSQL_DB", "customer_pulse")
    return create_engine(f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}")
# ── V1–V28 → business category map ────────────────────────────────────────────
V_CATEGORY_MAP = {
    "V1":  "travel", "V2":  "dining", "V3":  "retail",
    "V4":  "utilities", "V5": "entertainment", "V6": "groceries",
    "V7":  "fuel", "V8":  "healthcare", "V9":  "subscriptions",
    "V10": "insurance", "V11": "education", "V12": "dining",
    "V13": "travel", "V14": "retail", "V15": "fuel",
    "V16": "groceries", "V17": "entertainment", "V18": "utilities",
    "V19": "healthcare", "V20": "subscriptions", "V21": "retail",
    "V22": "dining", "V23": "travel", "V24": "groceries",
    "V25": "fuel", "V26": "entertainment", "V27": "utilities",
    "V28": "insurance",
}

def assign_category(row):
    """Pick category from the V column with the highest absolute value."""
    v_cols = [c for c in row.index if c.startswith("V")]
    dominant = row[v_cols].abs().idxmax()
    return V_CATEGORY_MAP.get(dominant, "other")

# ── Download datasets ──────────────────────────────────────────────────────────
def download_datasets():
    os.makedirs("data/raw", exist_ok=True)
    datasets = [
        ("mlg-ulb/creditcardfraud",                  "creditcardfraud.zip"),
        ("shantanudhakadd/bank-customer-churn-prediction", "bank-customer-churn-prediction.zip"),
    ]
    for slug, fname in datasets:
        out_path = f"data/raw/{fname}"
        if not os.path.exists(out_path):
            logger.info(f"Downloading {slug}...")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug, "-p", "data/raw/"],
                check=True
            )
        else:
            logger.info(f"Already exists: {out_path} — skipping download")

        # Unzip
        with zipfile.ZipFile(out_path, "r") as z:
            z.extractall("data/raw/")
            logger.info(f"Extracted: {out_path}")

# ── Load creditcard.csv → transactions ────────────────────────────────────────
def load_transactions(engine):
    path = "data/raw/creditcard.csv"
    logger.info(f"Reading {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows from creditcard.csv")

    # Assign synthetic customer IDs (cycle over 10,000 unique customers)
    np.random.seed(42)
    n_customers = 10_000
    df["customer_id"] = [f"CUST_{str(i % n_customers).zfill(5)}" for i in range(len(df))]

    # Assign synthetic dates spread across 2022–2024
    date_range = pd.date_range("2022-01-01", "2024-12-31", periods=len(df))
    df["txn_date"] = date_range.date

    # Map V1–V28 to business category
    logger.info("Assigning transaction categories from V-features...")
    df["category"] = df.apply(assign_category, axis=1)

    # Merchant type derived from Amount quartile
    df["merchant_type"] = pd.cut(
        df["Amount"],
        bins=[0, 10, 50, 200, float("inf")],
        labels=["micro", "small", "mid", "large"],
        right=True
    ).astype(str)

    # Select and rename columns to match transactions schema
    txn_df = df[["customer_id", "txn_date", "Amount", "category", "merchant_type"]].copy()
    txn_df.rename(columns={"Amount": "amount"}, inplace=True)
    txn_df["amount"] = txn_df["amount"].round(2)
    txn_df["customer_id"] = txn_df["customer_id"].map(str)
    txn_df["category"] = txn_df["category"].map(str)
    txn_df["merchant_type"] = txn_df["merchant_type"].map(str)

    # Save processed copy
    os.makedirs("data/processed", exist_ok=True)
    txn_df.to_csv("data/processed/transactions_clean.csv", index=False)
    logger.info("Saved data/processed/transactions_clean.csv")

    # Bulk insert into MySQL
    logger.info("Inserting transactions into MySQL...")
    txn_df.to_sql("transactions", engine, if_exists="append", index=False, chunksize=5000)
    logger.info(f"Inserted {len(txn_df):,} rows into transactions table ✅")
    return len(txn_df)

# ── Load churn dataset ─────────────────────────────────────────────────────────
def load_churn(engine):
    # Try both common filenames from this Kaggle dataset
    for fname in ["Churn_Modelling.csv", "churn.csv", "bank_churn.csv"]:
        path = f"data/raw/{fname}"
        if os.path.exists(path):
            break
    else:
        logger.warning("Churn CSV not found — check data/raw/ for the correct filename")
        return 0

    logger.info(f"Reading churn dataset from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows")

    # Standardise column names to lowercase
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Extract fields needed for model training later
    churn_df = df[["customerid", "tenure", "balance", "exited"]].copy()
    churn_df.rename(columns={
        "customerid": "customer_id",
        "exited":     "churn_flag"
    }, inplace=True)
    churn_df["customer_id"] = churn_df["customer_id"].astype(str).apply(
        lambda x: f"CUST_{x.zfill(5)}"
    )

    # Save processed copy
    churn_df.to_csv("data/processed/churn_clean.csv", index=False)
    logger.info("Saved data/processed/churn_clean.csv")
    logger.info(f"Churn dataset ready — {len(churn_df):,} rows, churn rate: "
                f"{churn_df['churn_flag'].mean():.1%}")
    return len(churn_df)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("Kaggle Loader Started")
    logger.info("=" * 60)

    download_datasets()
    engine = get_engine()

    txn_count   = load_transactions(engine)
    churn_count = load_churn(engine)

    logger.info(f"Done — transactions: {txn_count:,}, churn rows: {churn_count:,}")
    print(f"\n✅ Transactions loaded: {txn_count:,}")
    print(f"✅ Churn dataset ready: {churn_count:,} rows (in data/processed/)")

if __name__ == "__main__":
    main()
