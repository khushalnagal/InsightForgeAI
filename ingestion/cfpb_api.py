import os
import json
import requests
import mysql.connector
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
logger.add("logs/cfpb_ingestion.log", rotation="1 MB", retention="7 days", level="INFO")

# ── MySQL connection ───────────────────────────────────────────────────────────
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB", "customer_pulse")
    )

# ── Get last successful run date ───────────────────────────────────────────────
def get_last_run_date(cursor):
    cursor.execute("""
        SELECT last_run_at FROM pipeline_runs
        WHERE pipeline_name = 'cfpb' AND status = 'success'
        ORDER BY last_run_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    if row:
        last_run = row[0]
        logger.info(f"Last successful run found: {last_run}")
        return last_run.strftime("%Y-%m-%d")
    else:
        logger.info("No previous run found — defaulting to 2020-01-01")
        return "2020-01-01"

# ── Fetch one page from CFPB API ───────────────────────────────────────────────
def fetch_page(date_min, frm, size=1000):
    url = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
    params = {
        "date_received_min": date_min,
        "format": "json",
        "size": size,
        "frm": frm,
        "no_aggs": "true",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Encoding": "identity"   # Disable compression to avoid chunking issues
    }

    logger.info(f"Fetching page frm={frm}, date_min={date_min}")

    raw_content = b""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=60, stream=True)
        if response.status_code != 200:
            logger.warning(f"Stopping pagination at frm={frm} due to API limit")
            return []

        # ── Stream + cap download at 10MB per page ────────────────────────────
        MAX_BYTES = 10 * 1024 * 1024  # 10MB cap
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                raw_content += chunk
                if len(raw_content) >= MAX_BYTES:
                    response.close()
                    break

        logger.info(f"Downloaded {len(raw_content)} bytes for frm={frm}")

    except requests.exceptions.ChunkedEncodingError:
        logger.warning(f"ChunkedEncodingError at frm={frm} — attempting to parse partial data")

    # ── Parse JSON ────────────────────────────────────────────────────────────
    try:
        data = json.loads(raw_content)
        # Response is either {"hits": {"hits": [...]}} or a plain array [...]
        if isinstance(data, list):
            hits = data
        elif isinstance(data, dict):
            hits = data.get("hits", {}).get("hits", [])
        else:
            hits = []
        logger.info(f"Parsed {len(hits)} records from page frm={frm}")
        return hits

    except json.JSONDecodeError:
        logger.warning(f"Full JSON parse failed at frm={frm} — extracting objects manually")
        return extract_objects_manually(raw_content, size)

# ── Manual JSON object extractor (fallback) ────────────────────────────────────
def extract_objects_manually(raw_content, limit):
    text = raw_content.decode("utf-8", errors="ignore")
    records = []
    depth = 0
    current = ""
    inside = False

    for char in text:
        if char == "{":
            depth += 1
            inside = True
        if inside:
            current += char
        if char == "}":
            depth -= 1
            if depth == 0 and inside:
                try:
                    obj = json.loads(current)
                    records.append(obj)
                    current = ""
                    inside = False
                    if len(records) >= limit:
                        break
                except json.JSONDecodeError:
                    current = ""
                    inside = False

    logger.info(f"Manually extracted {len(records)} objects")
    return records

# ── Parse a single hit into a DB row tuple ─────────────────────────────────────
def parse_record(item):
    src = item.get("_source", item)  # Handle both formats

    # Parse date safely
    date_raw = src.get("date_received")
    try:
        date_received = datetime.strptime(date_raw[:10], "%Y-%m-%d").date() if date_raw else None
    except Exception:
        date_received = None

    return (
        src.get("complaint_id"),
        date_received,
        src.get("product"),
        src.get("sub_product"),
        src.get("issue"),
        src.get("complaint_what_happened") or src.get("consumer_complaint_narrative"),
        src.get("company"),
        src.get("state"),
        src.get("company_response"),
        src.get("consumer_disputed")
    )

# ── Upsert records into MySQL ──────────────────────────────────────────────────
def upsert_records(cursor, records):
    query = """
        INSERT INTO complaints 
            (id, date_received, product, sub_product, issue, narrative, 
             company, state, company_response, consumer_disputed)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            date_received       = VALUES(date_received),
            product             = VALUES(product),
            sub_product         = VALUES(sub_product),
            issue               = VALUES(issue),
            narrative           = VALUES(narrative),
            company             = VALUES(company),
            state               = VALUES(state),
            company_response    = VALUES(company_response),
            consumer_disputed   = VALUES(consumer_disputed)
    """
    cursor.executemany(query, records)
    logger.info(f"Upserted {cursor.rowcount} rows")
    return cursor.rowcount

# ── Log pipeline run to pipeline_runs ─────────────────────────────────────────
def log_pipeline_run(cursor, records_loaded, status):
    cursor.execute("""
        INSERT INTO pipeline_runs (pipeline_name, last_run_at, records_loaded, status)
        VALUES ('cfpb', %s, %s, %s)
    """, (datetime.now(), records_loaded, status))
    logger.info(f"pipeline_runs updated — status={status}, records={records_loaded}")

# ── Main ingestion loop ────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("CFPB Ingestion Pipeline Started")
    logger.info("=" * 60)

    conn = get_db_connection()
    cursor = conn.cursor()
    logger.info("MySQL connected ✅")

    # Get last run date for incremental loading
    date_min = get_last_run_date(cursor)
    logger.info(f"Pulling complaints from: {date_min}")

    total_inserted = 0
    frm = 0
    page = 1
    PAGE_SIZE = 1000

    try:
        while True:
            logger.info(f"── Page {page} (frm={frm}) ──────────────────────")
            hits = fetch_page(date_min=date_min, frm=frm, size=PAGE_SIZE)

            if not hits:
                logger.info("No more records — pagination complete ✅")
                break

            # Parse all hits into DB rows
            rows = []
            for item in hits:
                try:
                    row = parse_record(item)
                    if row[0]:  # Only add if complaint_id exists
                        rows.append(row)
                except Exception as e:
                    logger.warning(f"Skipping malformed record: {e}")

            if not rows:
                logger.info("No valid rows on this page — stopping")
                break

            # Upsert into MySQL
            upsert_records(cursor, rows)
            conn.commit()
            total_inserted += len(rows)

            logger.info(f"Page {page} done — {len(rows)} records upserted | Total so far: {total_inserted}")

            # If we got fewer records than page size, we're on the last page
            if len(hits) < PAGE_SIZE:
                logger.info("Last page reached ✅")
                break

            frm += PAGE_SIZE
            page += 1

        # Log successful run
        log_pipeline_run(cursor, total_inserted, "success")
        conn.commit()
        logger.info(f"Pipeline completed successfully ✅ | Total records: {total_inserted}")
        print(f"\n✅ Done! Inserted/updated {total_inserted} records into MySQL")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        log_pipeline_run(cursor, total_inserted, "failed")
        conn.commit()
        print(f"\n❌ Pipeline failed: {e}")

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()
