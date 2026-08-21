"""backend/es_client.py -- shared Elasticsearch client singleton. Ported
from ui/app.py:296-300 (get_es_client)."""

from elasticsearch import Elasticsearch

from . import config

_client = None


def get_es_client():
    global _client
    if _client is None:
        _client = Elasticsearch(config.ES_HOST)
    return _client
