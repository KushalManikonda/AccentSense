import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -------------------------------------------------------
# GLOBAL MONGO PATCH — applied before any module imports
# db.mongo so tests never make real network connections.
# -------------------------------------------------------
_mongo_patcher = patch("pymongo.MongoClient", return_value=MagicMock())
_mongo_patcher.start()


# -----------------------------
# PASSING TEST FIXTURES
# -----------------------------

@pytest.fixture
def valid_audio():
    return np.random.randn(16000).astype(np.float32)


@pytest.fixture
def valid_embedding():
    return np.random.randn(1, 768)


@pytest.fixture
def valid_prediction():
    return {
        "label": "telugu",
        "confidence": 0.95
    }


# -----------------------------
# FAILURE TEST FIXTURES
# -----------------------------

@pytest.fixture
def invalid_audio_bytes():
    return b"not_real_audio"


@pytest.fixture
def empty_audio():
    return np.array([], dtype=np.float32)


@pytest.fixture
def invalid_embedding():
    return np.random.randn(1, 100)


# -----------------------------
# EDGE CASE FIXTURES
# -----------------------------

@pytest.fixture
def short_audio():
    return np.random.randn(100).astype(np.float32)


@pytest.fixture
def long_audio():
    return np.random.randn(16000 * 60).astype(np.float32)


@pytest.fixture
def silent_audio():
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def borderline_confidence():
    return {
        "label": "kannada",
        "confidence": 0.5001
    }


@pytest.fixture
def stereo_audio():
    return np.random.randn(16000, 2).astype(np.float32)