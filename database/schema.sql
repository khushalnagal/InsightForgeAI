CREATE DATABASE customer_pulse;
USE customer_pulse;

CREATE TABLE complaints (
id VARCHAR(64) PRIMARY KEY,
date_received DATE,
product VARCHAR(128),
sub_product VARCHAR(128),
issue VARCHAR(256),
narrative TEXT,
company VARCHAR(256),
state VARCHAR(100),
company_response VARCHAR(256),
consumer_disputed VARCHAR(8),
loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
INDEX idx_date (date_received),
INDEX idx_product (product),
INDEX idx_state (state)
);

CREATE TABLE transactions (
txn_id INT AUTO_INCREMENT PRIMARY KEY,
customer_id VARCHAR(64),
txn_date DATE,
amount DECIMAL(12,2),
category VARCHAR(64),
merchant_type VARCHAR(64),
INDEX idx_customer (customer_id),
INDEX idx_txn_date (txn_date)
);

CREATE TABLE behavioral_segments (
customer_id VARCHAR(64) PRIMARY KEY,
segment_label VARCHAR(64),
cluster_id TINYINT,
churn_probability DECIMAL(5,4),
churn_flag TINYINT DEFAULT 0,
last_scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
INDEX idx_segment (segment_label),
INDEX idx_churn_flag (churn_flag)
);

CREATE TABLE credit_signals (
customer_id VARCHAR(64) PRIMARY KEY,
shap_top_features JSON,
reason_card TEXT,
updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE pipeline_runs (
run_id INT AUTO_INCREMENT PRIMARY KEY,
pipeline_name VARCHAR(64),
last_run_at TIMESTAMP,
records_loaded INT,
status VARCHAR(16)
);
