# S3 Explorer

A multi-user Streamlit app to browse S3 buckets efficiently.

AWS access uses the **local credential chain** (environment variables,
`~/.aws/credentials`, EC2/ECS instance role, …) — users never enter AWS
credentials. App users sign in with a username + password, and their **role**
controls what they can do.

## Features

- **Multi-user with roles**
  - `read-only` — browse, filter, preview, presigned URLs, download
  - `write` — read-only **+** upload, delete (with confirmation), create
    folders, rename/move objects
  - `admin` — write **+** in-app user management (add/delete users, change
    roles, reset passwords) **+** Firefly URL configuration

- **Bucket selection** — dropdown of all buckets the local credentials can
  list, plus free-text entry for public or cross-account buckets (optional
  unsigned mode for public buckets)

- **Efficient navigation** — folder-style browsing with breadcrumbs and
  server-side pagination (`list_objects_v2` continuation tokens); pagination
  arrows at both the top and bottom of the listing

- **Filename filter** — `starts with` mode filters server-side (efficient,
  spans all pages); `contains` mode filters the current page

- **Rich file preview** (right-hand panel, 100 KB limit)
  - CSV / TSV → interactive sortable dataframe
  - JSON / GeoJSON / IPYNB → pretty-printed with syntax highlighting
  - 50+ text formats rendered with syntax highlighting: Python, Bash, SQL,
    YAML, TOML, XML, HTML, CSS, JavaScript/TypeScript, Rust, Go, Kotlin,
    Markdown, INI/properties, `.env`, Dockerfile, HCL, and more
  - Binary files flagged cleanly

- **File actions** (fixed-width popover per file)
  - 👁 Preview in right-hand panel
  - 🔗 Presigned URL — one-click copy (configurable expiry)
  - ⬇ Download — direct browser download via presigned URL
  - 🔭 Send to Firefly — push data to the Caltech/IPAC
    [Firefly](https://github.com/Caltech-IPAC/firefly) viewer using
    `firefly_client.show_data`; returns a browser URL to open the viewer
    (supports FITS, IPAC tables, CSV, VOTable, region files, etc.)
  - Rename / move, delete (write/admin only)

- **Firefly integration** — uses `firefly_client.FireflyClient.make_client()`
  to establish a WebSocket channel to the Firefly server, then calls
  `show_data(url)` with a presigned (or public) S3 URL. The viewer URL is
  displayed in the preview panel for one-click open. Default server:
  `https://irsa.ipac.caltech.edu/irsaviewer` — configurable by admins in
  the sidebar.

## Run

```bash
cd s3-explorer
pip install -r requirements.txt
streamlit run app.py
```

### Docker

```bash
docker build -t s3-explorer .
docker run -p 8501:8501 \
  -v ~/.aws:/root/.aws:ro \
  s3-explorer
```

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit>=1.36` | UI framework |
| `boto3` / `botocore` | S3 client |
| `PyYAML` | `users.yaml` storage |
| `firefly_client>=3.4` | Firefly viewer integration |

## Admin

Manage users via the CLI (`auth.py`) without starting the app:

```bash
# list all users
python auth.py list

# change password (prompts if omitted)
python auth.py passwd admin
python auth.py passwd admin newpassword

# add user
python auth.py add alice --role write

# change role
python auth.py role alice read-only

# delete user
python auth.py delete alice
```

## Users

On first run a `users.yaml` is created next to `app.py` (git-ignored,
PBKDF2-hashed passwords) with two bootstrap accounts:

| user    | password | role      |
|---------|----------|-----------|
| `admin` | `admin`  | admin     |
| `roman` | `roman`  | read-only |

**Change both passwords immediately** from the *User management* tab or CLI.

## AWS permissions

The app only ever uses the permissions of the local credentials/assumed role.
App roles restrict users *below* that level, never above it. For full
functionality the local identity needs:

| Permission | Used for |
|---|---|
| `s3:ListAllMyBuckets` | bucket dropdown |
| `s3:ListBucket` | folder/file listing |
| `s3:GetObject` | preview, presigned URLs, download, Firefly |
| `s3:PutObject` | upload, create folder, rename/move |
| `s3:DeleteObject` | delete, rename/move |
