# Roman Documentation Navigator (Streamlit)

This repository branch contains a Streamlit app (`app.py`) for browsing GitHub documentation repositories and previewing the inferred main document for a selected repo/version.

## Features

- GitHub authentication using a personal access token
- Repository discovery with filtering by visibility, archive status, affiliation, topic, and search text
- Default focus on repositories tagged with `roman-docs`
- Version selection from tags and branches
- Automatic main document inference for repos matching `roman-ssc_d-[m|t|d]###`
- Inline rendering for Markdown and PDF, plus download support for PDF/DOCX/other files
- Built-in unit tests for core selection/filtering logic

## Requirements

- Python 3.10+
- `streamlit`
- `requests`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit requests
```

## Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL and provide a GitHub token with access to the repositories you want to browse.

## Token Sources (in priority order)

1. Sidebar token input
2. URL query parameter: `?token=...`
3. Environment variable: `GITHUB_TOKEN`

## Run Tests

```bash
python app.py --test
```
