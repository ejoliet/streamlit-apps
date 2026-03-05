# Roman Docs Portal

A minimal Streamlit application that authenticates with GitHub and shows documentation repositories whose names start with `roman-` and include the `roman-docs` topic. Selecting a repository displays its latest release or commit info and renders its primary markdown file.

## Prerequisites

- Python 3.9+
- GitHub OAuth app with the redirect URL pointing to where you will run Streamlit (defaults to `http://localhost:8501`)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Export the required environment variables from your GitHub OAuth app:
   ```bash
   export GITHUB_CLIENT_ID=your_client_id
   export GITHUB_CLIENT_SECRET=your_client_secret
   # Optional if you changed the redirect URL in the OAuth app:
   export GITHUB_REDIRECT_URI=http://localhost:8501
   ```

## Run

```bash
streamlit run app.py
```

You will be prompted to log in with GitHub. After approving the OAuth request, the UI lists repositories that match the filters on the left and displays their main markdown content on the right. Only repositories that start with `roman-` and include the `roman-docs` topic are shown.
