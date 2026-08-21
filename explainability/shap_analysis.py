import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from urllib.parse import quote_plus
from loguru import logger

load_dotenv()

password = quote_plus(os.getenv('MYSQL_PASSWORD'))
engine = create_engine(
    f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{password}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
)

feature_cols = ['avg_monthly_spend', 'spend_trend', 'spend_volatility',
                'days_since_last_txn', 'txn_count_90d',
                'high_value_txn_count', 'creditscore', 'age',
                'tenure', 'balance', 'numofproducts',
                'isactivemember', 'estimatedsalary']

def run_shap_analysis():
    logger.info("Loading churn model and feature data...")

    model = joblib.load('models/saved/churn_model.joblib')

    features = pd.read_csv('data/processed/customer_features.csv')
    churn_df = pd.read_csv('data/raw/Churn_Modelling.csv')

    churn_df = churn_df[['CustomerId', 'CreditScore', 'Age', 'Tenure',
                          'Balance', 'NumOfProducts', 'HasCrCard',
                          'IsActiveMember', 'EstimatedSalary', 'Exited']]
    churn_df.columns = [c.lower() for c in churn_df.columns]
    churn_df = churn_df.reset_index(drop=True)
    features = features.reset_index(drop=True)

    merged = features.copy()
    merged['creditscore'] = churn_df['creditscore']
    merged['age'] = churn_df['age']
    merged['tenure'] = churn_df['tenure']
    merged['balance'] = churn_df['balance']
    merged['numofproducts'] = churn_df['numofproducts']
    merged['isactivemember'] = churn_df['isactivemember']
    merged['estimatedsalary'] = churn_df['estimatedsalary']

    FEATURE_COLS = ['avg_monthly_spend', 'spend_trend', 'spend_volatility',
                'days_since_last_txn', 'txn_count_90d',
                'high_value_txn_count', 'creditscore', 'age',
                'tenure', 'balance', 'numofproducts',
                'isactivemember', 'estimatedsalary']
    X = merged[FEATURE_COLS].values
    X_df = pd.DataFrame(X, columns=FEATURE_COLS)

    logger.info("Running SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_df)

    # Global summary plot
    os.makedirs('explainability', exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_df, show=False)
    plt.tight_layout()
    plt.savefig('explainability/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("SHAP summary plot saved to explainability/shap_summary.png")

    # Per customer top 3 SHAP features for churn_flag=1 customers
    logger.info("Extracting top 3 SHAP features per at-risk customer...")
    shap_array = shap_values.values

    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT customer_id FROM behavioral_segments WHERE churn_flag = 1
        """))
        at_risk = [row[0] for row in result.fetchall()]

    logger.info(f"Processing {len(at_risk)} at-risk customers...")

    # Make sure credit_signals rows exist
    with engine.begin() as conn:
        for cid in at_risk:
            conn.execute(text("""
                INSERT IGNORE INTO credit_signals (customer_id)
                VALUES (:cid)
            """), {"cid": cid})

    # Map customer_id to index
    cid_to_idx = {row: i for i, row in enumerate(merged['customer_id'])}

    updated = 0
    with engine.begin() as conn:
        for cid in at_risk:
            idx = cid_to_idx.get(cid)
            if idx is None:
                continue
            customer_shap = shap_array[idx]
            top3_idx = np.argsort(np.abs(customer_shap))[-3:][::-1]
            top3 = {FEATURE_COLS[i]: round(float(customer_shap[i]), 4) for i in top3_idx}
            conn.execute(text("""
                UPDATE credit_signals
                SET shap_top_features = :shap
                WHERE customer_id = :cid
            """), {"shap": json.dumps(top3), "cid": cid})
            updated += 1

    logger.info(f"SHAP features written for {updated} customers")

    # Waterfall plots for 3 sample customers
    sample_customers = at_risk[:3]
    for i, cid in enumerate(sample_customers):
        idx = cid_to_idx.get(cid)
        if idx is None:
            continue
        plt.figure()
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.tight_layout()
        plt.savefig(f'explainability/waterfall_{cid}.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Waterfall plot saved for {cid}")

    logger.info("SHAP analysis complete.")

if __name__ == "__main__":
    run_shap_analysis()