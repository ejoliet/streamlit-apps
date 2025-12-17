# ASDF Streamlit Viewer

A lightweight **Streamlit** app to explore **ASDF (.asdf) files** interactively.

It automatically:
- Browses the ASDF tree
- Detects numeric arrays
- Recognizes **x vs y** data patterns
- Displays **tables** and **interactive plots**
- Falls back gracefully for large or complex structures

Ideal for astronomy/science ASDF files (Roman, JWST, generic ASDF).

---

## Features

- 📂 Upload and inspect `.asdf` files
- 🌳 Tree-based navigation of ASDF nodes
- 📊 Automatic x/y detection:
  - `(N,2)` or `(2,N)` numeric arrays
  - Dicts like `{x, y}`, `{time, flux}`, `{wavelength, flux}`
  - Structured NumPy arrays
- 📈 Interactive plots with real x-axes (Altair)
- 📋 Tabular previews for arrays and tables
- 🧠 Safe handling of large arrays (truncation + memmap support)

---

## Requirements

- Python **3.9+**
- macOS or Linux (Windows should work but not actively tested)

---

## Installation

Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```


## dependencies

```bash
pip install --upgrade streamlit asdf numpy pandas altair
```

## running

From the directory containing app.py:

```bash
streamlit run app.py
```

Your browser will open automatically at:

```bash
http://localhost:8501
```


## usage

- Upload an .asdf file using the sidebar
- Navigate the ASDF tree using the node selector
- When an array is detected:
Tables are shown automatically
x/y series are plotted interactively
- Large arrays are summarized safely

## Tips

- If your ASDF file uses custom extensions or tags, enable
“Ignore missing extensions” in the sidebar.
- Enable memmap for very large array-backed datasets.
- 1D arrays are plotted as index vs value automatically.

## Common x/y Patterns Detected

- array.shape == (N, 2) or (2, N)
- {"x": [...], "y": [...]}
- {"time": [...], "flux": [...]}
- Structured arrays with fields like x/y, time/flux


## License

MIT

## Future Ideas

- Manual x/y selector for arbitrary dicts
- WCS-aware image/cube visualization
- ASDF schema/tag inspection
- Export plots as PNG/CSV
