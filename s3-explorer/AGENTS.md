# S3 Explorer — agent notes

Streamlit multi-user S3 browser. Three files matter: `app.py` (all UI),
`auth.py` (users/roles + CLI), `firefly_connector.py` (Firefly wrapper).

## Run

```bash
pip install -r requirements.txt
streamlit run app.py          # http://localhost:8501
python auth.py list           # user management CLI (add/passwd/role/delete)
```

Quick sanity check after edits: `python -c "import ast; ast.parse(open('app.py').read())"`

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
- `users.yaml` is git-ignored, PBKDF2-hashed, bootstrap creds admin/admin —
  never commit it or weaken the hashing.

## Conventions

- Single-file UI: resist splitting `app.py` unless asked.
- Per-row widget keys are namespaced by S3 key (`f"prev_{row['key']}"`).
- Roles: `read-only` < `write` < `admin`; gate UI with `can_write` / `role`.
