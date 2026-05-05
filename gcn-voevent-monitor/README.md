# GCN VOEvent Live Monitor

A Streamlit dashboard that consumes real-time astronomical alerts from [NASA's General Coordinates Network (GCN)](https://gcn.nasa.gov) via Kafka and displays them with interactive charts and a sky map.

## Background

[GCN](https://gcn.nasa.gov) distributes machine-readable alerts for high-energy transient events — gamma-ray bursts (GRBs), gravitational-wave candidates, IceCube neutrino tracks, and more — from observatories such as Fermi, Swift, INTEGRAL, IceCube, and LIGO/Virgo. Alerts are encoded as [VOEvents](https://www.ivoa.net/documents/VOEvent/) (XML) and streamed over Apache Kafka.

This app started from the [example GCN Kafka consumer](https://gcn.nasa.gov/docs/client) (`consumer.py`) that subscribes to the full set of `gcn.classic.voevent.*` topics and prints raw messages. It was extended into a persistent, browsable dashboard by adding:

- A **SQLite database** (`db.py`) that accumulates events for up to one month and prunes older records automatically.
- A **background consumer thread** (`gcn_consumer.py`) that runs alongside the Streamlit process, parsing each VOEvent XML payload and writing structured records to the database.
- A **Streamlit UI** (`app.py`) with live metrics, an hourly event-rate chart, a per-topic bar chart, a RA/Dec sky scatter plot, an event table, and a raw XML viewer.

## Repository layout

Within the parent `streamlit-apps` mono-repo each subdirectory is a self-contained Streamlit application:

```
streamlit-apps/
├── gcn-voevent-monitor/   ← this app
│   ├── app.py
│   ├── consumer.py        (original GCN example consumer)
│   ├── gcn_consumer.py    (background thread wrapper)
│   ├── db.py
│   ├── requirements.txt
│   └── README.md
├── another-app/
│   └── ...
└── ...
```

## Prerequisites

- Python 3.11+
- A free GCN account — [register here](https://gcn.nasa.gov/login) to obtain a **client ID** and **client secret**.

## Installation

```bash
# 1. Clone the mono-repo and enter this app's folder
git clone git@github.com:ejoliet/streamlit-apps.git
cd streamlit-apps/gcn-voevent-monitor

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Configuration

Credentials are read from environment variables at startup. The easiest way is a local `.env` file (never committed):

```bash
cp .env.example .env
# edit .env and fill in your values
```

`.env` contents:

```
GCN_CLIENT_ID=your-client-id-here
GCN_CLIENT_SECRET=your-client-secret-here
```

Get your credentials from [gcn.nasa.gov/quickstart](https://gcn.nasa.gov/quickstart).

Alternatively export the variables in your shell before running:

```bash
export GCN_CLIENT_ID=your-client-id
export GCN_CLIENT_SECRET=your-client-secret
```

> `.env` and `*.db` are listed in `.gitignore` — they will never be included in commits.

## Running the app

```bash
streamlit run app.py
```

The browser opens at `http://localhost:8501`. The Kafka consumer starts automatically as a background thread; events accumulate in `gcn_events.db` (SQLite, created on first run).

### Dashboard features

| Section | Description |
|---|---|
| Metrics row | Event count for the selected time window, events with sky positions, total DB size, active topics |
| Hourly chart | Bar chart of event rate over the last 1–24 h (slider in sidebar) |
| Topic breakdown | Top-20 topics by event count |
| Sky map | RA/Dec scatter plot for events that carry positional information |
| Event table | Scrollable table with topic, role, IVORN, coordinates, description |
| Raw XML viewer | Full VOEvent XML for any selected event |

## Contributing

1. Fork the repo and create a feature branch from `main`:

   ```bash
   git checkout -b feat/gcn-voevent-<your-change>
   ```

2. Make your changes inside `gcn-voevent-monitor/`. Keep each Streamlit app self-contained — do not introduce cross-app dependencies.

3. Update `requirements.txt` if you add new dependencies.

4. Open a pull request against `main` on `ejoliet/streamlit-apps` with a clear description of what changed and why.

### Adding a new Streamlit app to the mono-repo

Create a new subdirectory at the repo root, add at minimum `app.py`, `requirements.txt`, and a `README.md`, then open a PR. No shared infrastructure is required.

## License

MIT
