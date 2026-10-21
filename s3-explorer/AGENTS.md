# S3 Explorer — agent notes

Streamlit multi-user S3 browser. Three files matter: `app.py` (all UI),
`auth.py` (users/roles + CLI), `firefly_connector.py` (Firefly wrapper).

## Run

```bash
uv venv && uv pip install -r requirements.txt   # no pyproject.toml; requirements.txt only
uv run streamlit run app.py   # http://localhost:8501
uv run auth.py list           # user management CLI (add/passwd/role/delete)
```

Quick sanity check after edits: `python -c "import ast; ast.parse(open('app.py').read())"`
No test suite exists; that parse check is the only automated gate.

Docker: `Dockerfile` runs with `--server.baseUrlPath=files`, so the container
serves at `http://localhost:8501/files` (not `/`). There is no `.dockerignore`;
`COPY . .` bakes `users.yaml`, `.venv/` and `.omc/` into the image. Add one
before publishing an image.

## Gotchas (learned the hard way)

- **Streamlit doesn't reload imported modules.** Editing
  `firefly_connector.py` (or `auth.py`) requires restarting the server;
  only `app.py` hot-reloads. Stale `@st.cache_resource` objects survive code
  edits too — `get_firefly()` has a `validate=` guard for this; keep it.
- **Firefly channel semantics:** a `show_data` dispatch sent before any
  browser tab is connected to the channel is silently lost (no replay).
  That's why each send goes out twice: WebSocket dispatch (tab open) + boot
  URL `?__action=app_data.externalUpload&url=…` (tab being opened). Don't
  "simplify" to one path.
- **Firefly URL action params arrive as strings** — `immediate=false` is
  truthy in JS. Preview-first is encoded by *omitting* `immediate`.
- **`components.html` iframes re-execute their script on remount**
  (popover open/close, reruns). The uuid nonce + localStorage guard in the
  Firefly block prevents duplicate `window.open` — keep it.
- **One viewer tab** is maintained via a named window
  (`firefly_{channel}`); cross-origin access throwing on the probe means
  the viewer is loaded there — that's expected, not an error.
- **Unsigned (public) buckets:** S3 rejects `ResponseContent*` params on
  anonymous presigned URLs; `st.session_state.anonymous` gates this.
- `users.yaml` is git-ignored, PBKDF2-hashed, bootstrap creds admin/admin
  and roman/roman (read-only) — never commit it or weaken the hashing.
- **Caching + fragments:** `list_page`, `presign`, `fetch_head` are
  `@st.cache_data` (ttl 60/30/60 s); `anonymous` is an explicit arg — never
  read `st.session_state` inside a cached fn. Each file row is
  `@st.fragment file_row`; the preview panel is `@st.fragment preview_panel`.
  Inside a fragment `st.rerun()` reruns only the fragment — anything that
  changes another panel needs `st.rerun(scope="app")`. Every S3 write must
  `st.cache_data.clear()` **then** `st.rerun()`; fragments mean nothing else
  forces a full rerun. The `↻` button clears all caches.

## Conventions

- Single-file UI: resist splitting `app.py` unless asked.
- Per-row widget keys are namespaced by S3 key (`f"prev_{row['key']}"`).
- Roles: `read-only` < `write` < `admin`; gate UI with `can_write` / `role`.
- Non-obvious decisions are logged one line each in `implementation-notes.md`.
