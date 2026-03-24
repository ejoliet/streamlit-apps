# DayLog — Local-first Journal

A local-first daily journaling app built with Streamlit. All data is stored in a local SQLite file — nothing leaves your machine unless you export it.

## Features

- **Daily entries** — mood, tasks, and free-form notes per day
- **Auto weather** — historical and forecast weather via [Open-Meteo](https://open-meteo.com/) (no API key required), auto-detected from your IP with manual city fallback
- **Photos** — attach images (JPG, PNG, WebP) stored directly in SQLite
- **Week view** — quick overview of the current week's entries
- **ICS export** — export any entry as a calendar event
- **Backup** — download a zipped copy of your database from the sidebar
- **Search** — full-text search across notes and tasks

## Requirements

- Python 3.9+
- pip

## Installation

```bash
# Clone the repo (or navigate to this directory)
cd daylog

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

## Configuration

| Environment variable | Default      | Description                                                     |
|----------------------|--------------|-----------------------------------------------------------------|
| `JOURNAL_DB_PATH`    | `journal.db` | Default SQLite database file path                               |
| `JOURNAL_DB_DIR`     | _(unset)_    | Folder of `.db` files; enables a database selector in the sidebar |

When `JOURNAL_DB_DIR` is set, the sidebar shows a dropdown of all `.db` files in that folder so you can switch journals without restarting the app. `JOURNAL_DB_PATH` is used as the default when no selection has been made.

Examples:

```bash
# Point to a specific database
JOURNAL_DB_PATH=~/Documents/my-journal.db streamlit run app.py

# Point to a folder of databases — selector appears in the sidebar
JOURNAL_DB_DIR=~/Documents/journals streamlit run app.py
```

## Data & Privacy

All data (entries, photos, weather cache) is stored locally in `journal.db` (SQLite). The only outbound network calls are:

- `ipapi.co` — one-time IP geolocation on startup (for auto weather location)
- `geocoding-api.open-meteo.com` — city geocoding when using manual fallback
- `api.open-meteo.com` — weather data fetch per day (cached locally after first fetch)

No account, login, or API key is required.

## Backup & Restore

**Backup:** click "Download backup (journal.db.zip)" in the sidebar.

**Restore:** replace `journal.db` (or the path set by `JOURNAL_DB_PATH`) with your backed-up file and restart the app.
