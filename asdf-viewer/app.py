import json
import tempfile
from typing import Any, List, Tuple

import numpy as np
import streamlit as st
import asdf


st.set_page_config(page_title="ASDF Viewer", layout="wide")
st.title("ASDF File Viewer")


# ----------------------------
# Helpers
# ----------------------------
def is_numeric_ndarray(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.number)

def is_1d_numeric(a: np.ndarray) -> bool:
    return is_numeric_ndarray(a) and a.ndim == 1 and a.size > 0

def as_1d_numeric(a: Any) -> np.ndarray | None:
    arr = try_as_ndarray(a)
    if arr is None:
        return None
    # Convert masked arrays etc. safely
    try:
        arr = np.asarray(arr)
    except Exception:
        return None
    if is_1d_numeric(arr):
        return arr
    return None

def detect_xy(value: Any) -> tuple[np.ndarray, np.ndarray, str] | None:
    """
    Try to infer (x, y) from a node.
    Returns (x, y, mode) or None.
    """
    # Case 1: 2D numeric array with 2 columns (N,2) or 2 rows (2,N)
    arr = try_as_ndarray(value)
    if arr is not None:
        arr = np.asarray(arr)
        if is_numeric_ndarray(arr) and arr.ndim == 2 and arr.size > 0:
            if arr.shape[1] == 2:
                return arr[:, 0], arr[:, 1], "array(N,2)"
            if arr.shape[0] == 2:
                return arr[0, :], arr[1, :], "array(2,N)"
        # Case 2: structured array with fields
        if isinstance(arr, np.ndarray) and arr.dtype.names:
            names = [n.lower() for n in arr.dtype.names]
            # common pairs
            candidates = [
                ("x", "y"),
                ("time", "flux"),
                ("wavelength", "flux"),
                ("freq", "amp"),
                ("frequency", "amplitude"),
            ]
            for a, b in candidates:
                if a in names and b in names:
                    x = arr[arr.dtype.names[names.index(a)]]
                    y = arr[arr.dtype.names[names.index(b)]]
                    x = np.asarray(x).ravel()
                    y = np.asarray(y).ravel()
                    if is_1d_numeric(x) and is_1d_numeric(y) and len(x) == len(y):
                        return x, y, f"struct({a},{b})"

    # Case 3: dict containing x/y-like keys of 1D numeric arrays
    if isinstance(value, dict):
        # exact keys first
        key_pairs = [
            ("x", "y"),
            ("time", "flux"),
            ("wavelength", "flux"),
            ("frequency", "amplitude"),
            ("freq", "amp"),
        ]
        lower_map = {str(k).lower(): k for k in value.keys()}
        for a, b in key_pairs:
            if a in lower_map and b in lower_map:
                xa = as_1d_numeric(value[lower_map[a]])
                yb = as_1d_numeric(value[lower_map[b]])
                if xa is not None and yb is not None and xa.shape[0] == yb.shape[0]:
                    return xa, yb, f"dict({a},{b})"

    return None

def is_scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None


def summarize_value(x: Any) -> str:
    """Short human-friendly summary for the sidebar list."""
    try:
        if isinstance(x, dict):
            return f"dict ({len(x)} keys)"
        if isinstance(x, (list, tuple)):
            return f"list ({len(x)} items)"
        if isinstance(x, np.ndarray):
            return f"ndarray shape={x.shape}, dtype={x.dtype}"
        # Some ASDF arrays are lazy objects; many still stringify well:
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return f"array-like shape={getattr(x,'shape',None)}, dtype={getattr(x,'dtype',None)}"
        if is_scalar(x):
            return f"{type(x).__name__}: {x!r}"[:120]
        return f"{type(x).__name__}"
    except Exception:
        return f"{type(x).__name__}"


def get_by_path(root: Any, path: List[Any]) -> Any:
    cur = root
    for p in path:
        cur = cur[p]
    return cur


def enumerate_paths(root: Any, max_nodes: int = 5000) -> List[Tuple[List[Any], str]]:
    """
    Produce a list of (path, label) for selecting nodes.
    Path entries are dict keys (str) or list indices (int).
    """
    out: List[Tuple[List[Any], str]] = []
    stack: List[Tuple[List[Any], Any]] = [([], root)]
    visited = 0

    while stack and visited < max_nodes:
        path, node = stack.pop()
        visited += 1

        # Add the node itself (except the root, which is still useful)
        label = "/" + "/".join(str(p) for p in path) if path else "/"
        label = f"{label}  —  {summarize_value(node)}"
        out.append((path, label))

        # Descend
        if isinstance(node, dict):
            for k in reversed(list(node.keys())):
                stack.append((path + [k], node[k]))
        elif isinstance(node, (list, tuple)):
            # limit list expansion a bit
            for i in reversed(range(min(len(node), 200))):
                stack.append((path + [i], node[i]))

    return out


def to_jsonable(x: Any, max_list: int = 200) -> Any:
    """
    Convert common objects to JSON-friendly structures for st.json.
    Avoid dumping huge arrays/lists.
    """
    if is_scalar(x):
        return x
    if isinstance(x, dict):
        # cap keys
        items = list(x.items())[:max_list]
        return {str(k): to_jsonable(v, max_list=max_list) for k, v in items}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v, max_list=max_list) for v in list(x)[:max_list]]
    if isinstance(x, np.ndarray):
        # show only small arrays
        if x.size <= 2000:
            return x.tolist()
        return {"__ndarray__": True, "shape": list(x.shape), "dtype": str(x.dtype), "note": "Too large to display fully"}
    # fallback
    return {"__type__": type(x).__name__, "__repr__": repr(x)[:2000]}


def try_as_ndarray(x: Any) -> np.ndarray | None:
    """Best-effort conversion for array-like nodes."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "__array__"):
        try:
            return np.asarray(x)
        except Exception:
            return None
    return None


# ----------------------------
# UI controls
# ----------------------------
with st.sidebar:
    st.header("Open options")

    ignore_missing_extensions = st.checkbox(
        "Ignore missing extensions",
        value=True,
        help="Helpful if the file references custom tags/extensions that aren't installed."
    )
    validate_checksums = st.checkbox("Validate checksums", value=False)
    memmap = st.checkbox(
        "Memory-map arrays (memmap)",
        value=False,
        help="May reduce RAM for large arrays if supported."
    )

    st.divider()
    uploaded = st.file_uploader("Upload an .asdf file", type=["asdf"])

    st.caption("Tip: if your ASDF file uses mission-specific tags, install the corresponding extension package(s).")


# ----------------------------
# Load file
# ----------------------------
if not uploaded:
    st.info("Upload an .asdf file to begin.")
    st.stop()

# Streamlit gives an UploadedFile (bytes in memory). Write to a temp file so asdf.open can seek efficiently.
with tempfile.NamedTemporaryFile(suffix=".asdf", delete=False) as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name

# Open ASDF (context manager is recommended)
# asdf.open supports flags like validate_checksums, memmap, ignore_missing_extensions. :contentReference[oaicite:2]{index=2}
try:
    with asdf.open(
        tmp_path,
        validate_checksums=validate_checksums,
        memmap=memmap,
        ignore_missing_extensions=ignore_missing_extensions,
    ) as af:
        tree = af.tree  # official access pattern :contentReference[oaicite:3]{index=3}

        # Build node list
        nodes = enumerate_paths(tree)
        labels = [lbl for _, lbl in nodes]

        col_left, col_right = st.columns([1, 2], gap="large")

        with col_left:
            st.subheader("Tree browser")
            selected_idx = st.selectbox("Select a node", range(len(labels)), format_func=lambda i: labels[i])
            sel_path = nodes[selected_idx][0]
            st.code("/" + "/".join(str(p) for p in sel_path) if sel_path else "/", language="text")

        value = get_by_path(tree, sel_path)

        with col_right:
            st.subheader("Node inspector")

            # Scalars
            if is_scalar(value):
                st.write("**Value**")
                st.write(value)

            # Dict / list
            elif isinstance(value, (dict, list, tuple)):
                # If it looks like dict(x,y), show plot/table first
                xy = detect_xy(value)
                if xy is not None:
                    x, y, mode = xy
                    st.success(f"Detected x/y series ({mode})")
                    df = {"x": x, "y": y}
                    st.dataframe(df, use_container_width=True)
                    st.line_chart(df)  # Streamlit will plot both columns; x is index-like here
                else:
                    st.write("**Preview (JSON-ish, truncated)**")
                    st.json(to_jsonable(value))

            # Arrays (and array-like)
            else:
                xy = detect_xy(value)
                arr = try_as_ndarray(value)
                if xy is not None:
                    x, y, mode = xy
                    st.success(f"Detected x/y series ({mode})")

                    # Table
                    df = {"x": x, "y": y}
                    st.dataframe(df, use_container_width=True)

                    # Chart: x vs y (true x-axis) using Altair
                    import pandas as pd
                    import altair as alt

                    pdf = pd.DataFrame(df)
                    chart = alt.Chart(pdf).mark_line().encode(
                        x=alt.X("x:Q", title="x"),
                        y=alt.Y("y:Q", title="y"),
                        tooltip=["x", "y"]
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)

                elif arr is None:
                    st.write("**Preview (repr)**")
                    st.code(repr(value)[:4000], language="text")

                else:
                    arr = np.asarray(arr)

                    # If it's a 1D numeric array, offer index vs value plot/table
                    if is_1d_numeric(arr):
                        st.info("1D numeric array: plotting index vs value (or choose custom x if you have one).")
                        n = arr.shape[0]
                        x = np.arange(n)
                        y = arr

                        st.dataframe({"index": x, "value": y}, use_container_width=True)

                        import pandas as pd
                        import altair as alt
                        pdf = pd.DataFrame({"x": x, "y": y})
                        chart = alt.Chart(pdf).mark_line().encode(
                            x=alt.X("x:Q", title="index"),
                            y=alt.Y("y:Q", title="value"),
                            tooltip=["x", "y"]
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)

                    # If it's 2D and numeric, show a table preview
                    elif is_numeric_ndarray(arr) and arr.ndim == 2 and arr.size > 0:
                        st.write("**2D array view (first 200x200 max)**")
                        st.dataframe(arr[:200, :200], use_container_width=True)

                    else:
                        st.write("**Array summary**")
                        st.write({"shape": arr.shape, "dtype": str(arr.dtype)})
                        st.json(to_jsonable(arr))


except Exception as e:
    st.error("Failed to open/parse ASDF.")
    st.exception(e)
    st.stop()

