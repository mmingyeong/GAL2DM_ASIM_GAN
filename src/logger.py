"""
logger.py

Description:
    Logging utility for unified console and file output across all modules.

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-06-10

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started.")

License:
    For academic use only. Contact the author before redistribution.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import logging
import os

def get_logger(name: str, log_dir: str = None, filename: str = None) -> logging.Logger:
    """
    Create or retrieve a logger with console + optional file logging.

    - If log_dir is None, use $DM2ICS_LOGDIR or ~/__dm2ics_model_benchmark/logs
    - Avoid duplicate handlers across repeated calls
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent duplicate prints via root

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # ---- Console handler (add once) ----
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                      for h in logger.handlers)
    if not has_console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # ---- Resolve log_dir (even when None) ----
    resolved_dir = log_dir or os.environ.get("DM2ICS_LOGDIR")
    if resolved_dir is None:
        home_dir = os.path.expanduser("~")
        resolved_dir = os.path.join(home_dir, "_dm2ics_model_benchmark", "logs")

    # ---- File handler (add once per file path) ----
    try:
        os.makedirs(resolved_dir, exist_ok=True)
        if filename is None:
            filename = f"{name}.log"
        file_path = os.path.join(resolved_dir, filename)

        # Do not add duplicate FileHandler pointing to the same file
        has_file = any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(file_path)
                       for h in logger.handlers)
        if not has_file:
            fh = logging.FileHandler(file_path, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    except PermissionError:
        logger.warning(f"⚠️ Cannot write to log_dir: {resolved_dir} (Permission Denied)")
    except OSError as e:
        logger.warning(f"⚠️ File logging disabled ({e.__class__.__name__}: {e})")

    return logger
