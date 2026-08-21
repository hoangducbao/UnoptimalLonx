"""submission -- CodaBench submission batch tooling for the OpticaLynx
(Routing101) pipeline.

Runs a folder of organizer query text files (*-kis.txt / *-qa.txt /
*-trake.txt) through the in-process retrieval pipeline (Mixed mode, and
Mixed-backed TRAKE) and writes the CodaBench CSV output for each query.

This package reuses the FastAPI backend's search modules directly (single
process — one set of ~4 GB model weights — mirroring backend/main.py's
lifespan warmup), so it must be run from the repo root with the repo's data
and models available.
"""

__version__ = "0.1.0"