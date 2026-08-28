import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

# -------------------------------------------------------------------
# Lazy proxy — MongoClient is created on first actual DB operation.
# This prevents DNS/SRV lookup at import time, which caused crashes
# during app startup and test collection.
# -------------------------------------------------------------------
_client = None

def _get_client():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client

class _LazyCollection:
    """Proxies all attribute/method access to the real pymongo Collection,
    but defers creating MongoClient until the first method call."""
    def __init__(self, collection_name: str):
        self._name = collection_name
        self._col = None

    def _resolve(self):
        if self._col is None:
            self._col = _get_client()["accentsense"][self._name]
        return self._col

    def __getattr__(self, attr):
        return getattr(self._resolve(), attr)

    def __getitem__(self, key):
        return self._resolve()[key]


cuisine_collection = _LazyCollection("cuisines")
history_collection = _LazyCollection("accent_history")
users_collection   = _LazyCollection("users")