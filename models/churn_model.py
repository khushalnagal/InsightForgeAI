import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import joblib
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
from loguru import logger

load_dotenv()

password = quote_plus(os.getenv('MYSQL_PASSWORD'))
engine = create_engine(
    f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{password}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
)

def run_churn_model():
    logger.info("Loading churn dataset...")

    # Load the Kaggle churn dataset
    churn_df = pd.read_csv('data/raw/Churn_Modelling.csv')
    logger.info(f"Churn dataset shape: {churn_df.shape}")

    # Load engineered features
    features = pd.read_csv('data/processed/customer_features.csv')

    # Prepare churn dataset - keep only useful columns
    churn_df = churn_df[['CustomerId', 'CreditScore', 'Age', 'Tenure',
                          'Balance', 'NumOfProducts', 'HasCrCard',
                          'IsActiveMember', 'EstimatedSalary', 'Exited']]
    churn_df.columns = [c.lower() for c in churn_df.columns]
    churn_df['customerid'] = 'CUST_' + churn_df['customerid'].astype(str).str.zfill(5)

    # Merge churn labels with engineered features on index position
    # since customer IDs are synthetic we align by position
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
    merged['churn'] = churn_df['exited']

    logger.info(f"Merged dataset shape: {merged.shape}")
    logger.info(f"Churn rate: {merged['churn'].mean():.2%}")

    # Define features for model training
    feature_cols = ['avg_monthly_spend', 'spend_trend', 'spend_volatility',
                'days_since_last_txn', 'txn_count_90d',
                'high_value_txn_count', 'creditscore', 'age',
                'tenure', 'balance', 'numofproducts',
                'isactivemember', 'estimatedsalary']

    X = merged[feature_cols].values
    y = merged['churn'].values

    # SMOTE + model in one pipeline so resampling is refit inside each CV
    # fold instead of once on the whole dataset — doing it beforehand lets
    # synthetic points derived from a held-out customer leak into that
    # customer's own training fold and inflates the CV score.
    logger.info("Building SMOTE + XGBoost pipeline...")
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42
        ))
    ])

    # Cross-validate on the ORIGINAL (imbalanced) data — SMOTE runs fresh
    # inside each fold via the pipeline
    logger.info("Training with 5-fold cross validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=['roc_auc', 'precision', 'recall'],
        return_train_score=False
    )

    logger.info(f"CV AUC:       {cv_results['test_roc_auc'].mean():.4f} +/- {cv_results['test_roc_auc'].std():.4f}")
    logger.info(f"CV Precision: {cv_results['test_precision'].mean():.4f}")
    logger.info(f"CV Recall:    {cv_results['test_recall'].mean():.4f}")

    # Fit final pipeline on full original data (SMOTE applied once here,
    # which is fine since this fit is not used to estimate performance)
    pipeline.fit(X, y)
    model = pipeline.named_steps["model"]

    # Score all customers with threshold 0.65 for recall optimization
    churn_probs = pipeline.predict_proba(X)[:, 1]
    churn_flags = (churn_probs >= 0.65).astype(int)

    merged['churn_probability'] = churn_probs
    merged['churn_flag'] = churn_flags

    logger.info(f"Customers flagged as churn risk: {churn_flags.sum()} ({churn_flags.mean():.2%})")

    # Write churn scores back to behavioral_segments table
    logger.info("Writing churn scores to MySQL...")
    with engine.begin() as conn:
        for _, row in merged[['customer_id', 'churn_probability', 'churn_flag']].iterrows():
            conn.execute(text("""
                UPDATE behavioral_segments
                SET churn_probability = :prob,
                    churn_flag = :flag
                WHERE customer_id = :cid
            """), {
                "prob": float(row['churn_probability']),
                "flag": int(row['churn_flag']),
                "cid": row['customer_id']
            })

    logger.info("Churn scores written to behavioral_segments")

    # Save model
    os.makedirs('models/saved', exist_ok=True)
    joblib.dump(model, 'models/saved/churn_model.joblib')
    logger.info("Churn model saved to models/saved/churn_model.joblib")

    return merged

if __name__ == "__main__":
    df = run_churn_model()
    print(df[['customer_id', 'churn_probability', 'churn_flag']].head(10))
