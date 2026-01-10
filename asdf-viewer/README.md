# Roman ASDF Viewer with jdaviz

A **Streamlit** app for viewing **Nancy Grace Roman Space Telescope** ASDF data files and displaying images with **jdaviz Imviz**.

---

## Features

- 🔭 **Roman-specific support** via `roman_datamodels`
- 📊 **Automatic detection** of Roman image arrays (`data`, `dq`, `err`, etc.)
- 📋 **Metadata display** (instrument, detector, exposure type)
- 🌳 **Tree browser** for exploring the full ASDF structure
- 📋 **Array data display** — view arrays as tables with statistics
- 📈 **1D array charts** — automatic line plots for 1D data
- 🖼️ **2D array images** — matplotlib preview with percentile scaling
- 🔭 **jdaviz Imviz** — optional interactive image visualization
- 📈 **X/Y series detection** — auto-detect time/flux, wavelength/flux patterns

---

## Roman Data Structure

Roman ASDF files follow a standard structure:

```
root
└── roman
    ├── meta
    │   ├── instrument (name, detector, optical_element)
    │   ├── exposure (type, time, etc.)
    │   └── observation (program, etc.)
    ├── data          # Science image (2D or 3D)
    ├── dq            # Data quality flags
    ├── err           # Error array
    ├── var_poisson   # Poisson variance
    └── coverage      # Coverage map
```

---

## Requirements

- Python **3.10+**
- macOS or Linux

---

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install streamlit roman_datamodels jdaviz astropy numpy matplotlib
```

### Package Details

| Package | Purpose |
|---------|---------|
| `roman_datamodels` | Read Roman ASDF files with proper schema support |
| `jdaviz` | Interactive astronomical image visualization |
| `streamlit` | Web application framework |
| `astropy` | FITS file handling |
| `numpy` | Array operations |
| `matplotlib` | Quick preview plots |

---

## Running

```bash
cd asdf-viewer
streamlit run app.py
```

Open your browser at: `http://localhost:8501`

**Note:** The app is configured to support ASDF files up to **600MB** in size (see `.streamlit/config.toml`).

---

## Usage

### 1. Upload a Roman ASDF File

Upload a `.asdf` file from Roman using the sidebar file uploader.

### 2. View Overview

The **Overview** tab shows:
- Detected image arrays (data, dq, err, etc.)
- Roman metadata (instrument, detector, exposure type)

### 3. Browse Data Structure

The **Data Browser** tab lets you:
- Navigate the full ASDF tree
- Inspect individual nodes with full details
- **1D arrays** — displayed as interactive line charts with table view
- **2D arrays** — displayed as tables with optional image view
- **3D arrays** — slice selector with table and image view
- **X/Y series** — auto-detected (time/flux, wavelength/flux, etc.)

### 4. Visualize with Imviz

The **Imviz Viewer** tab:
1. Select an image array from the dropdown
2. For 3D arrays, select a frame/slice
3. View array data as table with statistics
4. See matplotlib preview (always available)
5. Optionally enable **jdaviz Imviz** for interactive viewing:
   - Pan, zoom, colormap adjustment
   - Region selection and analysis
   - Full astronomical visualization tools

---

## Roman Data Arrays

| Path | Description |
|------|-------------|
| `roman/data` | Science image data (DN/s or MJy/sr) |
| `roman/dq` | Data quality flags (bitmask) |
| `roman/err` | Error array |
| `roman/var_poisson` | Poisson variance |
| `roman/var_rnoise` | Read noise variance |
| `roman/coverage` | Coverage/weight map |
| `roman/dark_slope` | Dark current rate (reference files) |

---

## Accessing Roman Data Programmatically

Based on `roman_datamodels` documentation:

```python
import roman_datamodels as rdm

# Open a Roman ASDF file
dm = rdm.open('roman_image.asdf')

# Access metadata
print(dm.meta.instrument.detector)
print(dm.meta.exposure.type)

# Access data arrays
print(dm.data.shape)   # Science data
print(dm.dq.shape)     # Data quality
print(dm.err.shape)    # Error array

# Modify and save
dm.data[100, 100] = 42
dm.to_asdf('modified.asdf')
```

---

## Displaying Images with jdaviz

Based on `jdaviz` documentation:

```python
from jdaviz import Imviz
import numpy as np

# Create Imviz instance
imviz = Imviz()

# Load numpy array directly
arr = np.arange(100).reshape((10, 10))
imviz.load_data(arr, data_label='my_image')

# Display
imviz.show()
```

---

## Troubleshooting

### "Extension not installed" warning

This appears when `roman_datamodels` is not installed. Install it:

```bash
pip install roman_datamodels
```

### jdaviz stays at "Loading app"

1. **Wait longer** — large images (4000×4000) take 30+ seconds
2. **Open in new tab** — click the "Open in new tab" link
3. **Check terminal** — look for error messages in the terminal running streamlit
4. **Try different port** — change the port number if 8930 is in use

### Can't open ASDF file

Ensure you have the correct version of `roman_datamodels` matching your file:

```bash
pip install --upgrade roman_datamodels
```

---

## Example Roman Files

Roman ASDF files can be obtained from:
- [MAST Archive](https://mast.stsci.edu/) (when available)
- Roman Science Operations Center
- Simulated data from `romanisim`

---

## License

MIT

---

## References

- [roman_datamodels documentation](https://roman-datamodels.readthedocs.io/)
- [jdaviz documentation](https://jdaviz.readthedocs.io/)
- [ASDF format specification](https://asdf-standard.readthedocs.io/)
- [Nancy Grace Roman Space Telescope](https://roman.gsfc.nasa.gov/)
