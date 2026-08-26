"""
backend/routes/facets.py -- exposes the structured metadata facets
(backend/metadata_filter.py) so the frontend can build the "Metadata
filter" dropdown without hardcoding subject/province lists that would go
stale as more lots get extracted into pipeline/*.csv.
"""

from fastapi import APIRouter

from .. import metadata_filter as md

router = APIRouter()


@router.get("/api/facets")
def get_facets():
    return md.get_facets()
