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
    roles, reset passwords)
- **Bucket selection** — dropdown of all buckets the local credentials can
  list, plus free-text entry for public or cross-account buckets (with an
  optional unsigned mode for public buckets)
- **Efficient navigation** — folder-style browsing with breadcrumbs and
  server-side pagination (`list_objects_v2` continuation tokens)
- **Filename filter** — `starts with` mode filters server-side (efficient,
  spans all pages); `contains` mode filters the current page
- **ASCII preview** — files detected as UTF-8/ASCII text are previewed inline
  (first 100 KB); binary files are flagged instead
- **Presigned URLs** — generate a presigned GET URL for any file with one
  click and copy it from the code block (configurable expiry)

## Run

```bash
cd s3-explorer
pip install -r requirements.txt
streamlit run app.py
```

## Users

On first run a `users.yaml` is created next to `app.py` (git-ignored,
PBKDF2-hashed passwords) with two bootstrap accounts:

| user    | password | role      |
|---------|----------|-----------|
| `admin` | `admin`  | admin     |
| `roman` | `roman`  | read-only |

**Change both passwords immediately** from the *User management* tab (sign in
as `admin`).

## AWS permissions

The app only ever uses the permissions of the local credentials/assumed role.
App roles can restrict users *below* that level, never above it. For the full
feature set the local identity needs `s3:ListAllMyBuckets`, `s3:ListBucket`,
`s3:GetObject`, and (for write users) `s3:PutObject` / `s3:DeleteObject`.
