# backend/db/database.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import sys # Used for printing debug messages to stderr

# --- Database URL Configuration ---
# This constructs the path to your SQLite database file.
# It will create 'vigilant_echo.db' in the root of your project directory.
# os.path.dirname(__file__) gives the directory of the current file (backend/db)
# os.path.join(..., "..", "..") navigates up two levels to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATABASE_FILE = "vigilant_echo.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, DATABASE_FILE)}"

print(f"INFO: Database will be created/accessed at: {SQLALCHEMY_DATABASE_URL}", file=sys.stderr)

# --- Database Engine Creation ---
# The 'engine' is what SQLAlchemy uses to talk to your SQLite database file.
# connect_args={"check_same_thread": False} is needed for SQLite when using it with FastAPI
# because FastAPI handles requests across different threads.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

# --- Session Local for Database Interactions ---
# SessionLocal is a class that will be used to create new database sessions.
# A 'session' is like a temporary workspace where you load objects from the database,
# make changes to them, and then save those changes back.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Base for Declarative Models ---
# declarative_base() is a factory function that returns a new base class.
# This 'Base' class will be inherited by all your SQLAlchemy models (like our 'Source' model).
# It tells SQLAlchemy that these Python classes correspond to database tables.
Base = declarative_base()

# --- Dependency for FastAPI (to get a database session for requests) ---
# This 'get_db' function is a FastAPI "dependency."
# FastAPI will call this function for each incoming request that needs database access.
# It creates a new session, yields it (passes it to your route function),
# and then ensures the session is closed afterward, even if errors occur.
def get_db():
    db = SessionLocal() # Create a new session
    try:
        yield db # Yield the session to the FastAPI route function
    finally:
        db.close() # Close the session when the request is done (or an error occurs)