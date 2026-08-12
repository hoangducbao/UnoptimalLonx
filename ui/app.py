"""
app.py — minimal Streamlit UI for the C1 baseline: text query in, ranked
keyframe results out. Click a result to open a popup with nearby frames
from the same video and a "search in this video only" box. No auth, no
persistence beyond the index.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import config  # noqa: E402
import retrieve  # noqa: E402

st.set_page_config(page_title="C1 Baseline — Video Moment Retrieval", layout="wide")
st.title("C1 Baseline — Video Moment Retrieval")

st.session_state.setdefault("selected", None)  # (video_id, global_id) | None
# Scoped-search results have to live in session_state, not a local variable
# gated on the form's one-shot submitted flag -- otherwise clicking "Open"
# on one of THOSE results triggers a rerun where the form's submit button
# is (correctly) not re-triggered, so the whole scoped-results section
# (including the very button just clicked) would fail to re-render at all,
# silently dropping the click. Persisting here means it keeps rendering,
# and stays associated with the video it was searched in.
st.session_state.setdefault("scoped_results", None)
st.session_state.setdefault("scoped_results_video_id", None)


@st.cache_resource
def get_backend(backend_key: str):
    return retrieve.load_backend(backend_key)


def render_results(results, key_prefix: str):
    if not results:
        st.info("No results — the index may be empty, or try a different query.")
        return
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
            if st.button("Open ▸", key=f"{key_prefix}_open_{r['global_id']}"):
                st.session_state.selected = (r["video_id"], r["global_id"])
                st.rerun()


with st.sidebar:
    st.subheader("Backend")
    backend_keys = list(config.BACKENDS.keys())
    backend = st.selectbox(
        "Embedding backend", options=backend_keys,
        index=backend_keys.index(config.DEFAULT_BACKEND),
        format_func=lambda k: config.BACKENDS[k]["label"],
    )

    frame_index, class_index, conn = get_backend(backend)

    st.subheader("Index status")
    st.write(f"Frame vectors: {frame_index.ntotal if frame_index else 0:,}")
    st.write(f"Object classes: {class_index.ntotal if class_index else 0:,}")
    text_rows = conn.execute("SELECT COUNT(*) AS n FROM keyframe_text").fetchone()["n"]
    st.write(f"OCR/caption rows: {text_rows:,}")

    top_k = st.slider("Results", 5, 100, 24)
    st.subheader("Fusion (experimental)")
    st.caption(
        "Class-leg cosine similarities are compressed for compound queries "
        "(e.g. 'Motorcycle' 0.63 vs 'Candle' 0.61) -- tune these if results look noisy."
    )
    class_top_k = st.slider("Matched classes to consider", 1, 50, config.DEFAULT_CLASS_TOP_K)
    class_weight = st.slider("Class-leg (object) weight", 0.0, 1.0, config.DEFAULT_LEG_WEIGHTS["class"])

leg_weights = {**config.DEFAULT_LEG_WEIGHTS, "class": class_weight}


@st.dialog("Frame detail", width="large")
def show_frame_detail(video_id: str, global_id: int, default_query: str):
    st.subheader(f"{video_id} — global_id {global_id}")

    nearby = retrieve.get_nearby_frames(conn, video_id, global_id)
    if nearby:
        st.caption("Nearby frames in this video")
        ncols = st.columns(len(nearby))
        for c, f in zip(ncols, nearby):
            with c:
                if f["thumbnail_path"]:
                    st.image(f["thumbnail_path"], use_container_width=True)
                frame_id = f"{f['n']:03d}" if f["n"] is not None else f["filename"]
                label = f"**{frame_id}**" if f["is_target"] else frame_id
                st.caption(label)

    st.markdown("**Search in this video only**")
    with st.form(key="video_scoped_search"):
        scoped_query = st.text_input("Query", value=default_query, key="scoped_query_input")
        submitted = st.form_submit_button("Search in this video only")
    if submitted and scoped_query:
        st.session_state.scoped_results = retrieve.search(
            scoped_query, backend=backend, frame_index=frame_index, class_index=class_index,
            conn=conn, top_k=top_k, class_top_k=class_top_k, leg_weights=leg_weights,
            video_id=video_id,
        )
        st.session_state.scoped_results_video_id = video_id

    if (
        st.session_state.scoped_results is not None
        and st.session_state.scoped_results_video_id == video_id
    ):
        render_results(st.session_state.scoped_results, key_prefix="scoped")

    if st.button("Close"):
        st.session_state.selected = None
        st.session_state.scoped_results = None
        st.session_state.scoped_results_video_id = None
        st.rerun()


query = st.text_input("Query", placeholder="e.g. một người đàn ông đang lái xe máy")

if query:
    results = retrieve.search(
        query, backend=backend, frame_index=frame_index, class_index=class_index, conn=conn,
        top_k=top_k, class_top_k=class_top_k, leg_weights=leg_weights,
    )
    render_results(results, key_prefix="main")

# Calling the @st.dialog-decorated function (re-)opens the popup. This has
# to be called unconditionally every run the selection is set (not just
# right after the "Open" click) -- widgets *inside* the dialog (the scoped
# search form, its own "Open" buttons) each trigger their own rerun, and
# the dialog only stays visible across those if it's re-invoked every time.
if st.session_state.selected:
    sel_video_id, sel_global_id = st.session_state.selected
    show_frame_detail(sel_video_id, sel_global_id, query or "")
