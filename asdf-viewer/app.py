"""
Roman ASDF Viewer with jdaviz Imviz

A Streamlit app for viewing Nancy Grace Roman Space Telescope ASDF files.
- Opens Roman ASDF files using roman_datamodels
- Detects 2D image arrays (data, dq, err, etc.)
- Displays images using jdaviz Imviz

Based on context7.com documentation for roman_datamodels and jdaviz.
"""

import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, List, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from astropy.io import fits

# Try to import roman_datamodels for proper Roman support
try:
    import roman_datamodels as rdm

    ROMAN_DATAMODELS_AVAILABLE = True
except ImportError:
    ROMAN_DATAMODELS_AVAILABLE = False
    import asdf

# Page config
st.set_page_config(
    page_title="Roman ASDF Viewer",
    page_icon="🔭",
    layout="wide",
)
st.title("🔭 Roman Space Telescope ASDF Viewer")


# ----------------------------
# Roman-specific helpers
# ----------------------------

# Common Roman data array paths to highlight
ROMAN_DATA_ARRAYS = [
    "roman/data",  # Science data
    "roman/dq",  # Data quality
    "roman/err",  # Error array
    "roman/var_poisson",  # Poisson variance
    "roman/var_rnoise",  # Read noise variance
    "roman/dark_slope",  # Dark current rate
    "roman/coverage",  # Coverage map
]


def is_numeric_ndarray(a: np.ndarray) -> bool:
    """Check if array is numeric numpy array."""
    return isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.number)


def is_1d_numeric(a: np.ndarray) -> bool:
    """Check if array is 1D numeric."""
    return is_numeric_ndarray(a) and a.ndim == 1 and a.size > 0


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


def as_1d_numeric(a: Any) -> np.ndarray | None:
    """Try to convert to 1D numeric array."""
    arr = try_as_ndarray(a)
    if arr is None:
        return None
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


def to_jsonable(x: Any, max_list: int = 200) -> Any:
    """Convert common objects to JSON-friendly structures."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        items = list(x.items())[:max_list]
        return {str(k): to_jsonable(v, max_list=max_list) for k, v in items}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v, max_list=max_list) for v in list(x)[:max_list]]
    if isinstance(x, np.ndarray):
        if x.size <= 2000:
            return x.tolist()
        return {"__ndarray__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    return {"__type__": type(x).__name__, "__repr__": repr(x)[:2000]}


def summarize_value(x: Any) -> str:
    """Short human-friendly summary for display."""
    try:
        if isinstance(x, dict):
            return f"dict ({len(x)} keys)"
        if isinstance(x, (list, tuple)):
            return f"list ({len(x)} items)"
        if isinstance(x, np.ndarray):
            return f"ndarray shape={x.shape}, dtype={x.dtype}"
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return f"array-like shape={getattr(x, 'shape', None)}, dtype={getattr(x, 'dtype', None)}"
        if isinstance(x, (str, int, float, bool)) or x is None:
            return f"{type(x).__name__}: {x!r}"[:120]
        
        # For roman_datamodels and similar objects, show type and key count
        type_name = type(x).__name__
        
        # Check for dict-like with keys
        if hasattr(x, "keys") and callable(getattr(x, "keys")):
            try:
                key_count = len(list(x.keys()))
                return f"{type_name} ({key_count} keys)"
            except Exception:
                pass
        
        # Check for _data attribute
        if hasattr(x, "_data") and isinstance(getattr(x, "_data", None), dict):
            try:
                key_count = len(x._data)
                return f"{type_name} ({key_count} fields)"
            except Exception:
                pass
        
        return f"{type_name}"
    except Exception:
        return f"{type(x).__name__}"


def get_by_path(root: Any, path: List[Any]) -> Any:
    """Navigate to a node by path."""
    cur = root
    for p in path:
        # Try dict-style access first
        try:
            cur = cur[p]
            continue
        except (KeyError, TypeError, IndexError):
            pass
        
        # Try attribute access
        if hasattr(cur, str(p)):
            cur = getattr(cur, str(p))
            continue
        
        # Try _data access for roman_datamodels
        if hasattr(cur, "_data") and isinstance(cur._data, dict):
            if p in cur._data:
                cur = cur._data[p]
                continue
        
        # Last resort - raise error
        raise KeyError(f"Cannot access '{p}' in {type(cur)}")
    
    return cur


def get_node_children(node: Any) -> List[Tuple[str, Any]]:
    """
    Get child nodes from various types of objects.
    Returns list of (key, value) pairs.
    """
    children = []
    
    # Standard dict
    if isinstance(node, dict):
        for k in node.keys():
            try:
                children.append((str(k), node[k]))
            except Exception:
                pass
        return children
    
    # List/tuple
    if isinstance(node, (list, tuple)):
        for i in range(min(len(node), 200)):
            try:
                children.append((str(i), node[i]))
            except Exception:
                pass
        return children
    
    # Skip numpy arrays and basic types
    if isinstance(node, np.ndarray):
        return children
    if isinstance(node, (str, int, float, bool, bytes)) or node is None:
        return children
    
    # roman_datamodels and similar objects - try multiple approaches
    
    # Approach 1: Check for _data attribute (common in roman_datamodels)
    if hasattr(node, "_data") and isinstance(getattr(node, "_data", None), dict):
        try:
            for k in node._data.keys():
                try:
                    val = getattr(node, k, None)
                    if val is None:
                        val = node._data.get(k)
                    children.append((str(k), val))
                except Exception:
                    pass
            return children
        except Exception:
            pass
    
    # Approach 2: Check for __dataclass_fields__ (dataclasses)
    if hasattr(node, "__dataclass_fields__"):
        try:
            for k in node.__dataclass_fields__.keys():
                try:
                    children.append((str(k), getattr(node, k)))
                except Exception:
                    pass
            return children
        except Exception:
            pass
    
    # Approach 3: Check for _fields (namedtuple-like)
    if hasattr(node, "_fields"):
        try:
            for k in node._fields:
                try:
                    children.append((str(k), getattr(node, k)))
                except Exception:
                    pass
            return children
        except Exception:
            pass
    
    # Approach 4: Check for keys() method (dict-like)
    if hasattr(node, "keys") and callable(getattr(node, "keys")):
        try:
            for k in node.keys():
                try:
                    children.append((str(k), node[k]))
                except Exception:
                    try:
                        children.append((str(k), getattr(node, k)))
                    except Exception:
                        pass
            if children:
                return children
        except Exception:
            pass
    
    # Approach 5: Check for items() method
    if hasattr(node, "items") and callable(getattr(node, "items")):
        try:
            for k, v in node.items():
                children.append((str(k), v))
            if children:
                return children
        except Exception:
            pass
    
    # Approach 6: Use __dict__ for regular objects (but filter out private attrs)
    if hasattr(node, "__dict__"):
        try:
            for k, v in node.__dict__.items():
                if not k.startswith("_"):
                    children.append((str(k), v))
            if children:
                return children
        except Exception:
            pass
    
    # Approach 7: Try dir() for objects with properties
    try:
        type_name = type(node).__name__
        # Only for objects that look like data models
        if any(x in type_name.lower() for x in ["model", "node", "meta", "wcs", "ref"]):
            for attr in dir(node):
                if not attr.startswith("_") and not callable(getattr(node, attr, None)):
                    try:
                        val = getattr(node, attr)
                        # Skip methods and callables
                        if not callable(val):
                            children.append((attr, val))
                    except Exception:
                        pass
    except Exception:
        pass
    
    return children


def enumerate_paths(root: Any, max_nodes: int = 10000) -> List[Tuple[List[Any], str]]:
    """Produce a list of (path, label) for selecting nodes."""
    out: List[Tuple[List[Any], str]] = []
    stack: List[Tuple[List[Any], Any]] = [([], root)]
    visited = 0
    seen_ids = set()  # Prevent circular references

    while stack and visited < max_nodes:
        path, node = stack.pop()
        visited += 1
        
        # Prevent circular references for complex objects
        try:
            node_id = id(node)
            if node_id in seen_ids and not isinstance(node, (str, int, float, bool, type(None))):
                continue
            seen_ids.add(node_id)
        except Exception:
            pass

        label = "/" + "/".join(str(p) for p in path) if path else "/"
        label = f"{label}  —  {summarize_value(node)}"
        out.append((path, label))

        # Get children and add to stack in reverse order
        children = get_node_children(node)
        for k, v in reversed(children):
            stack.append((path + [k], v))

    return out


def find_2d_arrays(
    root: Any, max_nodes: int = 10000
) -> List[Tuple[List[Any], str, tuple, str]]:
    """
    Find all 2D numeric arrays in the ASDF tree.
    Returns list of (path, label, shape, description).
    """
    results = []
    stack: List[Tuple[List[Any], Any]] = [([], root)]
    visited = 0
    seen_ids = set()

    # Roman array descriptions
    roman_descriptions = {
        "data": "Science Data",
        "dq": "Data Quality Flags",
        "err": "Error Array",
        "var_poisson": "Poisson Variance",
        "var_rnoise": "Read Noise Variance",
        "dark_slope": "Dark Current Rate",
        "coverage": "Coverage Map",
    }

    while stack and visited < max_nodes:
        path, node = stack.pop()
        visited += 1
        
        # Prevent circular references
        try:
            node_id = id(node)
            if node_id in seen_ids and not isinstance(node, (str, int, float, bool, type(None))):
                continue
            seen_ids.add(node_id)
        except Exception:
            pass

        # Check if this node is a 2D or 3D array
        arr = try_as_ndarray(node)
        if arr is not None:
            arr = np.asarray(arr)
            if is_numeric_ndarray(arr) and arr.ndim >= 2 and arr.size > 0:
                label = "/" + "/".join(str(p) for p in path) if path else "/"

                # Get description for Roman arrays
                last_key = str(path[-1]) if path else ""
                desc = roman_descriptions.get(last_key, "")

                # For 2D arrays, add directly
                if arr.ndim == 2:
                    results.append((path, label, arr.shape, desc))
                # For 3D arrays, offer each slice
                elif arr.ndim == 3:
                    results.append((path, f"{label} (3D)", arr.shape, f"{desc} - 3D cube"))

        # Get children using the shared function
        children = get_node_children(node)
        for k, v in children:
            stack.append((path + [k], v))

    return results


def get_roman_metadata(dm: Any) -> dict:
    """Extract key Roman metadata from datamodel."""
    meta = {}
    try:
        if hasattr(dm, "meta"):
            m = dm.meta
            # Instrument info
            if hasattr(m, "instrument"):
                meta["Instrument"] = getattr(m.instrument, "name", "N/A")
                meta["Detector"] = getattr(m.instrument, "detector", "N/A")
                meta["Optical Element"] = getattr(m.instrument, "optical_element", "N/A")
            # Exposure info
            if hasattr(m, "exposure"):
                meta["Exposure Type"] = getattr(m.exposure, "type", "N/A")
            # Target
            if hasattr(m, "target"):
                meta["Target"] = getattr(m.target, "catalog_name", "N/A")
            # Observation
            if hasattr(m, "observation"):
                meta["Program"] = getattr(m.observation, "program", "N/A")
    except Exception:
        pass
    return meta


def find_free_port() -> int:
    """Find an available port."""
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def write_array_to_fits(arr2d: np.ndarray, out_path: str):
    """Write 2D numpy array to FITS file for jdaviz."""
    hdu = fits.PrimaryHDU(arr2d)
    hdu.writeto(out_path, overwrite=True)


def check_server_ready(url: str, timeout: float = 1.0) -> bool:
    """Check if a server is responding at the given URL."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def launch_jdaviz_imviz(
    fits_path: str, host: str = "localhost", port: int | None = None
) -> tuple[subprocess.Popen, str]:
    """
    Launch jdaviz standalone web app (Imviz) pointed at a FITS file.
    Returns (process, url).
    """
    if port is None:
        port = find_free_port()

    cmd = [
        "jdaviz",
        "--layout=imviz",
        "--host=0.0.0.0",
        f"--port={port}",
        fits_path,
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    url = f"http://{host}:{port}/"
    return proc, url


# ----------------------------
# UI - Sidebar
# ----------------------------
with st.sidebar:
    st.header("📂 Roman ASDF Options")

    # Show roman_datamodels status
    if ROMAN_DATAMODELS_AVAILABLE:
        st.success("✅ roman_datamodels installed")
    else:
        st.warning("⚠️ roman_datamodels not installed")
        st.caption("Install with: `pip install roman_datamodels`")

    st.divider()

    uploaded = st.file_uploader("Upload a Roman .asdf file", type=["asdf"])

    st.divider()
    st.caption(
        """
    **Roman Data Arrays:**
    - `roman/data` — Science image
    - `roman/dq` — Data quality flags
    - `roman/err` — Error array
    - `roman/coverage` — Coverage map
    """
    )

# ----------------------------
# Load file
# ----------------------------
if not uploaded:
    st.info("👆 Upload a Roman Space Telescope ASDF file to begin.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### About this app
        
        View **Nancy Grace Roman Space Telescope** ASDF data files:
        
        1. **Browse** Roman data structure and metadata
        2. **Detect** 2D image arrays (data, dq, err, etc.)
        3. **Visualize** images with **jdaviz Imviz**
        """
        )

    with col2:
        st.markdown(
            """
        ### Roman Data Structure
        
        Roman ASDF files contain:
        - `roman.data` — Science data array
        - `roman.meta` — Metadata (instrument, exposure, etc.)
        - `roman.dq` — Data quality flags
        - `roman.err` — Error array
        """
        )

    st.divider()

    st.markdown(
        """
    ### Installation
    
    ```bash
    pip install roman_datamodels jdaviz streamlit astropy numpy matplotlib
    ```
    """
    )
    st.stop()

# Write uploaded file to temp location
with tempfile.NamedTemporaryFile(suffix=".asdf", delete=False) as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name

# Open ASDF file
try:
    # Use roman_datamodels if available for proper Roman support
    if ROMAN_DATAMODELS_AVAILABLE:
        dm = rdm.open(tmp_path)
        # Access the underlying tree for browsing
        if hasattr(dm, "_asdf"):
            tree = dm._asdf.tree
        else:
            tree = {"roman": dm}
        is_roman_model = True
    else:
        import asdf

        af = asdf.open(tmp_path, ignore_missing_extensions=True)
        tree = af.tree
        dm = tree.get("roman", tree)
        is_roman_model = False

    # Find all 2D arrays in the file
    arrays_2d = find_2d_arrays(tree)

    # Create tabs for different views
    tab_overview, tab_browser, tab_imviz = st.tabs(
        ["📊 Overview", "🌳 Data Browser", "🖼️ Imviz Viewer"]
    )

    # ----------------------------
    # Tab 1: Overview
    # ----------------------------
    with tab_overview:
        st.subheader("Roman Data File Overview")

        # Get Roman metadata
        roman_meta = get_roman_metadata(dm)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("2D Arrays Found", len(arrays_2d))

        with col2:
            st.metric("Instrument", roman_meta.get("Instrument", "N/A"))

        with col3:
            st.metric("Detector", roman_meta.get("Detector", "N/A"))

        # Show metadata
        if roman_meta:
            st.markdown("### Metadata")
            meta_cols = st.columns(4)
            for i, (key, val) in enumerate(roman_meta.items()):
                with meta_cols[i % 4]:
                    st.markdown(f"**{key}:** {val}")

        # Show detected arrays
        if arrays_2d:
            st.markdown("### Image Arrays")
            st.success(f"✅ Found {len(arrays_2d)} image array(s)")

            for path, label, shape, desc in arrays_2d:
                desc_text = f" — *{desc}*" if desc else ""
                st.markdown(f"- **`{label}`** shape: `{shape}`{desc_text}")
        else:
            st.warning("No 2D/3D numeric arrays found in this file.")

    # ----------------------------
    # Tab 2: Data Browser
    # ----------------------------
    with tab_browser:
        st.subheader("Roman Data Browser")

        # Simpler tree renderer - only show structure, click to see details
        def render_tree_node(node: Any, path: List[Any], depth: int = 0, max_depth: int = 15):
            """Recursively render a tree node with expandable children."""
            if depth > max_depth:
                st.caption("⚠️ Max depth reached")
                return
            
            # Get summary and children
            summary = summarize_value(node)
            node_name = str(path[-1]) if path else "root"
            full_path = "/" + "/".join(str(p) for p in path) if path else "/"
            children = get_node_children(node)
            
            # Simple scalars - display inline
            if isinstance(node, (str, int, float, bool)) or node is None:
                st.write(f"**{node_name}:** `{repr(node)}`")
                return
            
            # Nodes with children - use expander
            if len(children) > 0:
                # Only auto-expand root level
                with st.expander(f"📁 {node_name} — {summary}", expanded=(depth == 0)):
                    st.caption(f"Path: `{full_path}`")
                    
                    # Detect and show special types
                    type_name = type(node).__name__
                    
                    # Astropy Time object
                    if type_name == 'Time' or (hasattr(node, 'iso') and hasattr(node, 'format') and hasattr(node, 'scale')):
                        show_time_summary(node, full_path)
                    # Numpy array
                    elif isinstance(node, np.ndarray):
                        # Check if it's datetime64
                        if np.issubdtype(node.dtype, np.datetime64):
                            show_datetime_array(node, full_path)
                        else:
                            show_array_summary(node, full_path)
                    # Astropy Table/QTable
                    elif hasattr(node, 'colnames'):
                        show_table_summary(node, full_path)
                    # List - show inline if small and simple
                    elif isinstance(node, list):
                        if len(node) <= 10 and all(isinstance(x, (str, int, float, bool)) or x is None for x in node):
                            st.write("**List values:**")
                            for i, val in enumerate(node):
                                st.write(f"  [{i}]: `{repr(val)}`")
                        else:
                            st.info(f"List with {len(node)} items")
                    # Other complex objects
                    elif not isinstance(node, (dict, tuple)):
                        st.info(f"Type: {type_name}")
                    
                    # Show children
                    st.divider()
                    for i, (child_key, child_val) in enumerate(children[:50]):  # Limit to 50 for performance
                        render_tree_node(child_val, path + [child_key], depth + 1, max_depth)
                    
                    if len(children) > 50:
                        st.caption(f"... and {len(children) - 50} more items")
            else:
                # Leaf node - show summary
                st.write(f"**{node_name}:** {summary}")
        
        def show_array_summary(arr: np.ndarray, path_str: str):
            """Show a quick summary of an array."""
            try:
                arr = np.asarray(arr)
                st.write(f"**Shape:** {arr.shape}, **dtype:** {arr.dtype}")
                
                if is_numeric_ndarray(arr):
                    st.write(f"**Range:** {np.min(arr):.4g} to {np.max(arr):.4g}")
                
                # For 2D arrays, show mini preview
                if arr.ndim == 2 and is_numeric_ndarray(arr):
                    if st.checkbox(f"Preview image", key=f"preview_{path_str}"):
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(6, 6))
                        im = ax.imshow(arr[:500, :500], aspect='auto', origin='lower', cmap='viridis')
                        plt.colorbar(im, ax=ax, label="Value")
                        ax.set_title("Preview (first 500x500)")
                        st.pyplot(fig)
                        plt.close()
                        
                        st.info("💡 Go to **Imviz Viewer** tab for full interactive visualization")
            except Exception as e:
                st.caption(f"Could not display array: {e}")
        
        def show_table_summary(table_obj, path_str: str):
            """Show a summary of Astropy Table/QTable."""
            try:
                # Try to get column names
                if hasattr(table_obj, 'colnames'):
                    colnames = table_obj.colnames
                    st.write(f"**Columns ({len(colnames)}):** {', '.join(colnames)}")
                    
                    # Try to convert to pandas for nice display
                    try:
                        import pandas as pd
                        df = table_obj.to_pandas()
                        
                        with st.expander("📊 View Table Data", expanded=True):
                            st.dataframe(df, use_container_width=True)
                        
                        # Show column info
                        with st.expander("📋 Column Details", expanded=False):
                            for col in colnames:
                                col_data = table_obj[col]
                                if hasattr(col_data, 'shape'):
                                    st.write(f"- **{col}:** shape={col_data.shape}, dtype={col_data.dtype}")
                                else:
                                    st.write(f"- **{col}:** {type(col_data).__name__}")
                    except Exception as e:
                        # Fallback: show columns as text
                        st.write("**Column Information:**")
                        for col in colnames:
                            try:
                                col_data = table_obj[col]
                                if hasattr(col_data, 'shape') and hasattr(col_data, 'dtype'):
                                    st.write(f"- **{col}:** shape={col_data.shape}, dtype={col_data.dtype}")
                                else:
                                    st.write(f"- **{col}:** {summarize_value(col_data)}")
                            except Exception:
                                st.write(f"- **{col}**")
                else:
                    st.info("Table structure not readable")
            except Exception as e:
                st.caption(f"Could not display table: {e}")
        
        def show_time_summary(time_obj, path_str: str):
            """Show a summary of Astropy Time object."""
            try:
                # Show the time value(s)
                st.write("**Time Value(s):**")
                
                # Try different formats
                try:
                    # ISO format
                    iso_val = time_obj.iso
                    if hasattr(iso_val, '__iter__') and not isinstance(iso_val, str):
                        # Array of times
                        st.write(f"- **ISO format:** {len(iso_val)} time values")
                        with st.expander("Show all times", expanded=(len(iso_val) <= 10)):
                            for i, t in enumerate(list(iso_val)[:100]):
                                st.write(f"  [{i}]: {t}")
                            if len(iso_val) > 100:
                                st.caption(f"... and {len(iso_val) - 100} more")
                    else:
                        # Single time
                        st.write(f"- **ISO:** `{iso_val}`")
                except Exception:
                    pass
                
                # Try to show other useful formats
                try:
                    if hasattr(time_obj, 'mjd'):
                        st.write(f"- **MJD:** {time_obj.mjd}")
                except Exception:
                    pass
                
                try:
                    if hasattr(time_obj, 'jd'):
                        st.write(f"- **JD:** {time_obj.jd}")
                except Exception:
                    pass
                
                # Show format and scale
                if hasattr(time_obj, 'format'):
                    st.write(f"- **Format:** {time_obj.format}")
                if hasattr(time_obj, 'scale'):
                    st.write(f"- **Scale:** {time_obj.scale}")
                    
            except Exception as e:
                st.caption(f"Could not display time: {e}")
        
        def show_datetime_array(arr, path_str: str):
            """Show datetime64 array in readable format."""
            try:
                st.write(f"**Datetime Array:** {arr.shape[0]} value(s)")
                
                # Convert to readable strings
                import pandas as pd
                timestamps = pd.to_datetime(arr)
                
                if len(timestamps) <= 20:
                    for i, ts in enumerate(timestamps):
                        st.write(f"[{i}]: {ts}")
                else:
                    st.write("**First 20 values:**")
                    for i, ts in enumerate(timestamps[:20]):
                        st.write(f"[{i}]: {ts}")
                    st.caption(f"... and {len(timestamps) - 20} more")
                    
            except Exception as e:
                st.caption(f"Could not format datetime: {e}")
        
        # Render the tree
        st.subheader("📂 Data Tree Browser")
        st.caption("Click on folders (📁) to expand the tree structure. Large arrays can be viewed in the Imviz tab.")
        
        # Add search functionality
        search_term = st.text_input(
            "🔍 Search tree nodes and values",
            placeholder="Search by path, name, or value (e.g., 'meta', 'WFI', '2024')...",
            help="Type to filter nodes. Tree will auto-expand to show matches in context."
        )
        
        # Find matching paths if search is active
        matching_paths = set()
        paths_to_expand = set()
        
        if search_term:
            def find_matching_paths(node, path=[]):
                """Find all paths that match the search term."""
                results = []
                search_lower = search_term.lower()
                
                # Check path match
                path_str = "/" + "/".join(str(p) for p in path) if path else "/"
                path_match = search_lower in path_str.lower()
                value_match = False
                
                # Check value match
                try:
                    if isinstance(node, (str, int, float, bool)) or node is None:
                        if search_lower in str(node).lower():
                            value_match = True
                    elif isinstance(node, np.ndarray):
                        if search_lower in str(node.dtype).lower():
                            value_match = True
                        elif node.dtype.kind in ['U', 'S', 'O']:
                            flat = node.flatten()
                            for item in flat[:100]:
                                if search_lower in str(item).lower():
                                    value_match = True
                                    break
                    elif hasattr(node, '__class__'):
                        if search_lower in type(node).__name__.lower():
                            value_match = True
                except Exception:
                    pass
                
                if path_match or value_match:
                    results.append(tuple(path))
                
                # Recurse
                children = get_node_children(node)
                for child_key, child_val in children[:200]:
                    results.extend(find_matching_paths(child_val, path + [child_key]))
                
                return results
            
            matches = find_matching_paths(tree, [])
            
            if matches:
                st.success(f"🔍 Found **{len(matches)}** matches - tree auto-expanded below")
                matching_paths = set(matches)
                
                # Calculate all parent paths to expand
                for match_path in matches:
                    for i in range(len(match_path) + 1):
                        paths_to_expand.add(tuple(match_path[:i]))
            else:
                st.warning("No matches found. Showing full tree below.")
        
        # Enhanced tree renderer with search highlighting
        def render_tree_with_search(node: Any, path: List[Any], depth: int = 0, max_depth: int = 15):
            """Render tree with auto-expansion for search matches."""
            if depth > max_depth:
                st.caption("⚠️ Max depth reached")
                return
            
            summary = summarize_value(node)
            node_name = str(path[-1]) if path else "root"
            full_path = "/" + "/".join(str(p) for p in path) if path else "/"
            children = get_node_children(node)
            
            path_tuple = tuple(path)
            is_match = path_tuple in matching_paths
            should_expand = path_tuple in paths_to_expand or depth == 0
            
            # Scalars
            if isinstance(node, (str, int, float, bool)) or node is None:
                if is_match:
                    st.success(f"✨ **{node_name}:** `{repr(node)}`")
                else:
                    st.write(f"**{node_name}:** `{repr(node)}`")
                return
            
            # Nodes with children
            if len(children) > 0:
                display_name = f"✨ {node_name}" if is_match else node_name
                
                with st.expander(f"📁 {display_name} — {summary}", expanded=should_expand):
                    st.caption(f"Path: `{full_path}`")
                    if is_match:
                        st.info("🎯 This node matches your search")
                    
                    type_name = type(node).__name__
                    
                    if type_name == 'Time' or (hasattr(node, 'iso') and hasattr(node, 'format') and hasattr(node, 'scale')):
                        show_time_summary(node, full_path)
                    elif isinstance(node, np.ndarray):
                        if np.issubdtype(node.dtype, np.datetime64):
                            show_datetime_array(node, full_path)
                        else:
                            show_array_summary(node, full_path)
                    elif hasattr(node, 'colnames'):
                        show_table_summary(node, full_path)
                    elif isinstance(node, list):
                        if len(node) <= 10 and all(isinstance(x, (str, int, float, bool)) or x is None for x in node):
                            st.write("**List values:**")
                            for i, val in enumerate(node):
                                st.write(f"  [{i}]: `{repr(val)}`")
                        else:
                            st.info(f"List with {len(node)} items")
                    elif not isinstance(node, (dict, tuple)):
                        st.info(f"Type: {type_name}")
                    
                    st.divider()
                    for i, (child_key, child_val) in enumerate(children[:50]):
                        render_tree_with_search(child_val, path + [child_key], depth + 1, max_depth)
                    
                    if len(children) > 50:
                        st.caption(f"... and {len(children) - 50} more items")
            else:
                if is_match:
                    st.success(f"✨ **{node_name}:** {summary}")
                else:
                    st.write(f"**{node_name}:** {summary}")
        
        # Render the tree
        st.subheader("📂 Data Tree Browser")
        st.caption("Tree will auto-expand to show search matches in context")
        
        with st.container():
            render_tree_with_search(tree, [], depth=0, max_depth=15)
        

    # ----------------------------
    # Tab 3: Imviz Viewer
    # ----------------------------
    with tab_imviz:
        st.subheader("🖼️ Display Roman Image with jdaviz Imviz")

        if not arrays_2d:
            st.warning("No 2D/3D numeric arrays found in this file.")
            st.stop()

        st.markdown(
            """
        View Roman image data using **jdaviz Imviz** — the STScI interactive 
        visualization tool for astronomical images.
        """
        )

        # Select which array to display
        array_options = {}
        for path, label, shape, desc in arrays_2d:
            desc_text = f" ({desc})" if desc else ""
            key = f"{label} — shape: {shape}{desc_text}"
            array_options[key] = (path, shape)

        selected_array = st.selectbox(
            "Select an image array",
            options=list(array_options.keys()),
        )

        if selected_array:
            selected_path, selected_shape = array_options[selected_array]
            arr_value = get_by_path(tree, selected_path)
            arr_original = np.asarray(try_as_ndarray(arr_value))
            arr = arr_original.copy()

            # Handle 3D arrays - let user select slice
            if arr.ndim == 3:
                num_slices = arr.shape[0]
                if num_slices > 1:
                    st.info(f"3D array with {num_slices} frames. Select a slice to display.")
                    slice_idx = st.slider(
                        "Select frame/slice",
                        min_value=0,
                        max_value=num_slices - 1,
                        value=0,
                        key="imviz_slice"
                    )
                    arr = arr[slice_idx]
                else:
                    st.info(f"3D array with only 1 frame. Displaying frame 0.")
                    arr = arr[0]

            st.write(f"**Array info:** shape={arr.shape}, dtype={arr.dtype}")
            st.write(f"**Value range:** min={arr.min():.4g}, max={arr.max():.4g}")

            # ----------------------------
            # Array Data Display
            # ----------------------------
            st.markdown("### Array Data")
            
            col_table, col_stats = st.columns([2, 1])
            
            with col_table:
                with st.expander("📋 View as Table (first 200x200)", expanded=False):
                    st.dataframe(arr[:200, :200], use_container_width=True)
            
            with col_stats:
                st.markdown("**Statistics:**")
                finite_arr = arr[np.isfinite(arr)]
                if len(finite_arr) > 0:
                    st.write(f"- Mean: {np.mean(finite_arr):.4g}")
                    st.write(f"- Median: {np.median(finite_arr):.4g}")
                    st.write(f"- Std: {np.std(finite_arr):.4g}")
                    st.write(f"- Non-zero: {np.count_nonzero(arr)}")

            st.divider()

            # ----------------------------
            # Image Display Options
            # ----------------------------
            st.markdown("### Image Viewer")
            
            # Quick matplotlib preview (always shown)
            show_preview = st.checkbox("📊 Show matplotlib preview", value=True)
            
            if show_preview:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 8))
                finite_arr = arr[np.isfinite(arr)]
                if len(finite_arr) > 0:
                    vmin, vmax = np.percentile(finite_arr, [1, 99])
                else:
                    vmin, vmax = arr.min(), arr.max()
                im = ax.imshow(arr, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, label="Value")
                ax.set_title(f"Roman Image - shape {arr.shape}")
                st.pyplot(fig)
                plt.close()

            st.divider()
            
            # ----------------------------
            # jdaviz Imviz (optional)
            # ----------------------------
            st.markdown("### jdaviz Imviz (Interactive)")
            st.caption("Launch jdaviz for interactive pan, zoom, colormap adjustment, and analysis tools.")
            
            use_jdaviz = st.checkbox("🔭 Use jdaviz Imviz for interactive viewing", value=False)
            
            if use_jdaviz:
                port = st.number_input(
                    "jdaviz port",
                    min_value=1024,
                    max_value=65535,
                    value=8930,
                    step=1,
                )

                # Session state for jdaviz process
                if "jdaviz_proc" not in st.session_state:
                    st.session_state.jdaviz_proc = None
                    st.session_state.jdaviz_url = None
                    st.session_state.jdaviz_fits = None
                    st.session_state.jdaviz_ready = False

                col_start, col_stop, col_refresh = st.columns(3)

                with col_start:
                    if st.button("▶️ Start Imviz", type="primary"):
                        # Stop any existing process
                        if st.session_state.jdaviz_proc is not None:
                            try:
                                st.session_state.jdaviz_proc.terminate()
                                time.sleep(0.5)
                            except Exception:
                                pass

                        # Write the array to a temporary FITS file
                        with st.spinner("Writing FITS file..."):
                            tmp_dir = tempfile.mkdtemp(prefix="roman_imviz_")
                            fits_path = os.path.join(tmp_dir, "roman_image.fits")
                            write_array_to_fits(arr, fits_path)

                        # Launch jdaviz
                        with st.spinner("Starting jdaviz..."):
                            proc, url = launch_jdaviz_imviz(
                                fits_path=fits_path, host="localhost", port=int(port)
                            )

                            st.session_state.jdaviz_proc = proc
                            st.session_state.jdaviz_url = url
                            st.session_state.jdaviz_fits = fits_path
                            st.session_state.jdaviz_ready = False

                        st.rerun()

                with col_stop:
                    if st.button("⏹️ Stop Imviz"):
                        if st.session_state.jdaviz_proc is not None:
                            try:
                                st.session_state.jdaviz_proc.terminate()
                            except Exception:
                                pass
                        st.session_state.jdaviz_proc = None
                        st.session_state.jdaviz_url = None
                        st.session_state.jdaviz_fits = None
                        st.session_state.jdaviz_ready = False
                        st.rerun()

                with col_refresh:
                    if st.button("🔄 Refresh"):
                        st.rerun()

                # Display embedded jdaviz viewer
                if st.session_state.jdaviz_url:
                    server_ready = check_server_ready(st.session_state.jdaviz_url)
                    proc_running = (
                        st.session_state.jdaviz_proc is not None
                        and st.session_state.jdaviz_proc.poll() is None
                    )

                    if not proc_running:
                        st.error("❌ jdaviz process stopped unexpectedly.")
                        if st.session_state.jdaviz_proc:
                            try:
                                output = st.session_state.jdaviz_proc.stdout.read()
                                if output:
                                    st.code(output[:2000], language="text")
                            except Exception:
                                pass
                        st.session_state.jdaviz_proc = None
                        st.session_state.jdaviz_url = None
                    elif server_ready:
                        st.success(f"✅ Imviz ready at: {st.session_state.jdaviz_url}")
                        st.markdown(f"[🔗 Open in new tab]({st.session_state.jdaviz_url})")

                        components.iframe(
                            st.session_state.jdaviz_url,
                            height=700,
                            scrolling=True,
                        )
                    else:
                        st.warning("⏳ jdaviz is starting... (10-30 seconds for large images)")
                        st.caption(f"Server URL: {st.session_state.jdaviz_url}")

                        progress_bar = st.progress(0, text="Waiting for server...")
                        for i in range(15):
                            time.sleep(1)
                            progress_bar.progress(
                                int((i + 1) / 15 * 100), text=f"Checking server... ({i+1}/15)"
                            )
                            if check_server_ready(st.session_state.jdaviz_url):
                                progress_bar.progress(100, text="Server ready!")
                                time.sleep(0.5)
                                st.rerun()
                                break
                        else:
                            st.info("Server still loading. Click Refresh to try again.")
                else:
                    st.info("👆 Click 'Start Imviz' to launch the interactive viewer.")

    # Close file handle if using asdf directly
    if not ROMAN_DATAMODELS_AVAILABLE:
        af.close()

except Exception as e:
    st.error("❌ Failed to open Roman ASDF file.")
    st.exception(e)

    st.markdown(
        """
    ### Troubleshooting
    
    1. **Install roman_datamodels:**
       ```bash
       pip install roman_datamodels
       ```
    
    2. **Verify it's a Roman file** — the file should contain a `roman` key at the top level.
    
    3. **Check file integrity** — ensure the file isn't corrupted.
    """
    )
    st.stop()
