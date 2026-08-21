import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os
from loguru import logger

load_dotenv()

password = quote_plus(os.getenv('MYSQL_PASSWORD'))
engine = create_engine(
    f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{password}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
)

def engineer_features():
    logger.info("Starting feature engineering...")

    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM transactions"))
        transactions = pd.DataFrame(result.fetchall(), columns=result.keys())

    logger.info(f"Loaded {len(transactions)} transactions")

    transactions['txn_date'] = pd.to_datetime(transactions['txn_date'])
    transactions['amount'] = transactions['amount'].astype(float)
    today = transactions['txn_date'].max()

    monthly = transactions.copy()
    monthly['month'] = monthly['txn_date'].dt.to_period('M')
    monthly_spend = monthly.groupby(['customer_id', 'month'])['amount'].sum().reset_index()

    avg_monthly_spend = monthly_spend.groupby('customer_id')['amount'].mean().reset_index()
    avg_monthly_spend.columns = ['customer_id', 'avg_monthly_spend']

    recent_6m = monthly_spend[monthly_spend['month'] >= (today - pd.DateOffset(months=6)).to_period('M')]

    def calc_slope(group):
        if len(group) < 2:
            return 0
        x = np.arange(len(group))
        slope = np.polyfit(x, group['amount'].values.astype(float), 1)[0]
        return slope

    spend_trend = recent_6m.groupby('customer_id').apply(calc_slope).reset_index()
    spend_trend.columns = ['customer_id', 'spend_trend']

    spend_volatility = monthly_spend.groupby('customer_id')['amount'].agg(
        lambda x: x.std() / x.mean() if x.mean() != 0 else 0
    ).reset_index()
    spend_volatility.columns = ['customer_id', 'spend_volatility']

    top_category = transactions.groupby(['customer_id', 'category'])['amount'].sum().reset_index()
    top_category = top_category.loc[top_category.groupby('customer_id')['amount'].idxmax()][['customer_id', 'category']]
    top_category.columns = ['customer_id', 'top_category']

    days_since = transactions.groupby('customer_id')['txn_date'].max().reset_index()
    days_since['days_since_last_txn'] = (today - days_since['txn_date']).dt.days
    days_since = days_since[['customer_id', 'days_since_last_txn']]

    # Transaction count in last 90 days — recency and engagement signal
    cutoff_90d = today - pd.DateOffset(days=90)
    recent_txns = transactions[transactions['txn_date'] >= cutoff_90d]
    txn_count_90d = recent_txns.groupby('customer_id').size().reset_index()
    txn_count_90d.columns = ['customer_id', 'txn_count_90d']

    # High value transaction count — spending intensity signal
    high_value_txns = recent_txns[recent_txns['amount'] > 100]
    high_value_txn_count = high_value_txns.groupby('customer_id').size().reset_index()
    high_value_txn_count.columns = ['customer_id', 'high_value_txn_count']

    features = avg_monthly_spend \
        .merge(spend_trend, on='customer_id', how='left') \
        .merge(spend_volatility, on='customer_id', how='left') \
        .merge(top_category, on='customer_id', how='left') \
        .merge(days_since, on='customer_id', how='left') \
        .merge(txn_count_90d, on='customer_id', how='left') \
        .merge(high_value_txn_count, on='customer_id', how='left')

    features['txn_count_90d'] = features['txn_count_90d'].fillna(0)
    features['high_value_txn_count'] = features['high_value_txn_count'].fillna(0)
    features = features.fillna(0)

    features.to_csv('data/processed/customer_features.csv', index=False)
    logger.info(f"Feature engineering done. Shape: {features.shape}")
    logger.info(f"Sample:\n{features.head(3)}")

    return features

if __name__ == "__main__":
    df = engineer_features()
    print(df.head())