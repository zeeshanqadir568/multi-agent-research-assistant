"""
Project configuration.

This file contains all configurable values used across the project.
Instead of hardcoding values throughout the codebase, every module
imports settings from here.
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ==========================================================
# Chunking
# ==========================================================

# Number of characters in one chunk
CHUNK_SIZE = 500

# Number of overlapping characters
CHUNK_OVERLAP = 80

# ==========================================================
# Embedding Model
# ==========================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================================================
# Hybrid Retrieval
# ==========================================================

# Weight between dense and sparse retrieval.
#
# 1.0 = Dense only
# 0.0 = BM25 only
#
# 0.5 = Equal contribution
#
HYBRID_ALPHA = 0.5

# ==========================================================
# Search
# ==========================================================

DEFAULT_TOP_K = 5