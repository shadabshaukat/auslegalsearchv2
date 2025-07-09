"""
Database connection module for auslegalsearchv2.
- Connects to PostgreSQL 16 with pgvector extension enabled.
- Provides SQLAlchemy engine and session makers.
- Checks/creates vector extension if needed.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Print DB URL for troubleshooting
def print_db_url(db_url):
    print(f"DB_URL in use: {db_url}")

DB_HOST = os.environ.get('AUSLEGALSEARCH_DB_HOST', 'localhost')
DB_PORT = os.environ.get('AUSLEGALSEARCH_DB_PORT', '5432')
DB_USER = os.environ.get('AUSLEGALSEARCH_DB_USER', 'postgres')
DB_PASSWORD = os.environ.get('AUSLEGALSEARCH_DB_PASSWORD', '')
DB_NAME = os.environ.get('AUSLEGALSEARCH_DB_NAME', 'postgres')

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DB_URL_2 = f"postgresql+psycopg2://{DB_USER}:'******'@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print_db_url(DB_URL_2)

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

def ensure_pgvector():
    """
    Ensures the pgvector extension is installed in the database.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

# Call this at app startup
ensure_pgvector()

"""
Usage:
from db.connector import engine, SessionLocal
with SessionLocal() as session:
    ...
"""
