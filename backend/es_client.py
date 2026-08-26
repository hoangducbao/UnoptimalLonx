"""backend/es_client.py -- shared Elasticsearch client singleton. Ported
from ui/app.py:296-300 (get_es_client)."""

from elasticsearch import Elasticsearch

from . import config

_client = None


def get_es_client(force_new: bool = False):
    global _client
    if _client is None or force_new:
        host = str(config.ES_HOST).replace("localhost", "127.0.0.1")
        if not host.startswith("http"):
            host = f"http://{host}"
        _client = Elasticsearch(
            hosts=[host],
            request_timeout=5,
            verify_certs=False,
            ssl_show_warn=False,
        )
    return _client

