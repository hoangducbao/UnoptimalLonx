"""backend/ -- FastAPI rewrite of ui/app.py's Streamlit UI. See the plan at
C:\\Users\\semic\\.claude\\plans\\replicated-percolating-dragonfly.md (or the
project's own docs once ported) for the architecture.

Adds pipeline/ to sys.path here, in the package __init__, so it runs before
any backend.* submodule is imported -- backend/search/keyframe.py does
`import clip_encoder` (pipeline/clip_encoder.py) at module level, same
workaround ui/app.py:63-68 uses.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PIPELINE_DIR = _REPO_ROOT / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))
