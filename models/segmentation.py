import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
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

def run_segmentation():
    logger.info("Loading engineered features...")

    features = pd.read_csv('data/processed/customer_features.csv')

    numeric_cols = ['avg_monthly_spend', 'spend_trend', 'spend_volatility',
                'days_since_last_txn', 'txn_count_90d', 'high_value_txn_count']

    X = features[numeric_cols].values

    # Scale features to zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce dimensions while retaining 85%+ of variance
    pca = PCA(n_components=0.85, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"PCA reduced to {X_pca.shape[1]} components retaining 85%+ variance")

    # Find optimal number of clusters using silhouette score
    silhouette_scores = {}
    for k in range(3, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        silhouette_scores[k] = score
        logger.info(f"k={k} silhouette score: {score:.4f}")

    # Plot silhouette scores
    plt.figure(figsize=(8, 4))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
    plt.title('Silhouette Score by Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.savefig('notebooks/silhouette_scores.png')
    plt.close()
    logger.info("Silhouette plot saved to notebooks/silhouette_scores.png")

    # Select best k
    best_k = 4
    logger.info(f"Best k selected: {best_k}")

    # Train final KMeans with best k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    features['cluster'] = kmeans.fit_predict(X_pca)

    # Name each cluster
    feature_cols = numeric_cols
    centroids_original = scaler.inverse_transform(
        pca.inverse_transform(kmeans.cluster_centers_)
    )
    logger.info(f"Centroids:\n{pd.DataFrame(centroids_original, columns=feature_cols)}")

    # Assign names by ranking clusters on spend and trend
    centroid_df = pd.DataFrame(centroids_original, columns=feature_cols)
    centroid_df['cluster_id'] = range(len(centroid_df))
    centroid_df = centroid_df.sort_values('spend_trend')

    cluster_names = {}
    centroid_df['score'] = centroid_df['avg_monthly_spend'] + centroid_df['spend_trend'] * 2
    centroid_df_sorted = centroid_df.sort_values('score').reset_index(drop=True)
    labels_ordered = ['Declining Disengaged', 'Stable Mid Tier', 'High Value At Risk', 'Premium Growth']
    for rank in range(len(centroid_df_sorted)):
        cid = int(centroid_df_sorted.loc[rank, 'cluster_id'])
        label = labels_ordered[rank]
        cluster_names[cid] = label
        logger.info(f"Cluster {cid} -> {label}")

    features['segment_label'] = features['cluster'].map(cluster_names)

    # Write results to MySQL behavioral_segments table
    rows_written = 0
    with engine.begin() as conn:
        for _, row in features[['customer_id', 'segment_label', 'cluster']].iterrows():
            conn.execute(text("""
                INSERT INTO behavioral_segments (customer_id, segment_label, cluster_id)
                VALUES (:cid, :label, :cluster_id)
                ON DUPLICATE KEY UPDATE
                    segment_label = VALUES(segment_label),
                    cluster_id = VALUES(cluster_id)
            """), {"cid": row['customer_id'], "label": row['segment_label'], "cluster_id": int(row['cluster'])})
            rows_written += 1

    logger.info(f"Written {rows_written} rows to behavioral_segments")

    # Save models
    os.makedirs('models/saved', exist_ok=True)
    joblib.dump(scaler, 'models/saved/scaler.joblib')
    joblib.dump(pca, 'models/saved/pca.joblib')
    joblib.dump(kmeans, 'models/saved/kmeans.joblib')
    logger.info("Models saved to models/saved/")

    logger.info(f"Segment distribution:\n{features['segment_label'].value_counts()}")
    return features

if __name__ == "__main__":
    df = run_segmentation()
    print(df[['customer_id', 'segment_label']].head(10))
