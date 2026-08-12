"""
app.py — minimal Streamlit UI for the C1 baseline: text query in, ranked
keyframe results out. No auth, no persistence beyond the index.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import config  # noqa: E402
import retrieve  # noqa: E402

st.set_page_config(page_title="C1 Baseline — Video Moment Retrieval", layout="wide")
st.title("C1 Baseline — Video Moment Retrieval")


@st.cache_resource
def get_backend():
    return retrieve.load_backend()


frame_index, class_index, conn = get_backend()

with st.sidebar:
    st.subheader("Index status")
    st.write(f"Frame vectors: {frame_index.ntotal if frame_index else 0:,}")
    st.write(f"Object classes: {class_index.ntotal if class_index else 0:,}")
    text_rows = conn.execute("SELECT COUNT(*) AS n FROM keyframe_text").fetchone()["n"]
    st.write(f"OCR/caption rows: {text_rows:,}")
    top_k = st.slider("Results", 5, 100, 24)
    st.subheader("Fusion (experimental)")
    st.caption(
        "Class-leg cosine similarities are compressed for compound queries "
        "(e.g. 'Motorcycle' 0.63 vs 'Candle' 0.61) so it's down-weighted by "
        "default -- tune these if results still look noisy."
    )
    class_top_k = st.slider("Matched classes to consider", 1, 50, config.DEFAULT_CLASS_TOP_K)
    class_weight = st.slider("Class-leg weight", 0.0, 1.0, config.DEFAULT_LEG_WEIGHTS["class"])

query = st.text_input("Query", placeholder="e.g. một người đàn ông đang lái xe máy")

if query:
    leg_weights = {**config.DEFAULT_LEG_WEIGHTS, "class": class_weight}
    results = retrieve.search(
        query, frame_index=frame_index, class_index=class_index, conn=conn,
        top_k=top_k, class_top_k=class_top_k, leg_weights=leg_weights,
    )
    if not results:
        st.info("No results — the index may be empty, or try a different query.")
    else:
        st.caption(f"{len(results)} results")
        cols = st.columns(4)
        for i, r in enumerate(results):
            with cols[i % 4]:
                if r["thumbnail_path"]:
                    st.image(r["thumbnail_path"], use_container_width=True)
                else:
                    st.markdown("*(no thumbnail)*")
                ts = f"{r['pts_time']:.2f}s" if r["pts_time"] is not None else "unknown time"
                frame_id = f"{r['n']:03d}" if r["n"] is not None else r["filename"]
                st.markdown(f"**{r['video_id']}** frame {frame_id} @ {ts}")
                badges = " ".join(f"`{name}`" for name in r["signals"])
                st.caption(f"score={r['rrf_score']:.4f}  signals: {badges}")
