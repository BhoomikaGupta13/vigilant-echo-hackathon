# backend/logging_config.py

import logging
import os
from logging.handlers import RotatingFileHandler
import sys

# Define log file name and path (in the project root)
LOG_FILE_NAME = "vigilant_echo_security.log"
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Project root
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# --- Logger Configuration ---
def configure_logging():
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # Capture all messages at DEBUG level and above

    # --- Console Handler ---
    # This sends log messages to the console (where Uvicorn runs)
    console_handler = logging.StreamHandler(sys.stderr) # Send to standard error, like print(..., file=sys.stderr)
    console_handler.setLevel(logging.INFO) # Only show INFO messages and above in the console

    # Define console log format: [LEVEL] [YYYY-MM-DD HH:MM:SS,ms] [MODULE_NAME]: MESSAGE
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s - %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File Handler (Rotating File) ---
    # This sends log messages to a file.
    # RotatingFileHandler:
    #   - LOG_FILE_PATH: The file where logs will be written.
    #   - maxBytes: Maximum size of the log file before it's rotated (e.g., 5 MB).
    #   - backupCount: Number of old log files to keep (e.g., keep 3 old logs).
    # When current log file reaches maxBytes, it's renamed (e.g., vigilant_echo_security.log.1),
    # and a new empty vigilant_echo_security.log is created.
    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=5 * 1024 * 1024, # 5 MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG) # Write all DEBUG messages and above to the file

    # Define file log format: More detailed, including module, line number
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Prevent duplicate logs from propagating to the root logger if
    # other modules also configure their own handlers.
    logger.propagate = False

    print(f"INFO: Logging configured. Logs will be written to: {LOG_FILE_PATH}", file=sys.stderr)