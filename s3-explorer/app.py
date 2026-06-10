"""
S3 Explorer — multi-user S3 bucket browser.

AWS access uses the local credential chain (env vars, ~/.aws, instance/task
role) so no AWS login is required. App users authenticate with a username +
password (see auth.py); their role gates what they can do:

  - read-only : browse, filter, preview ASCII files, presigned URLs, download
  - write     : read-only + upload, delete, create folders, rename/move
  - admin     : write + user management
"""

import string
from datetime import datetime
from typing import Optional

import boto3
import streamlit as st
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

import auth

st.set_page_config(
    page_title="S3 Explorer",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PREVIEW_MAX_BYTES = 100_000  # read at most this much for ASCII preview


# ──────────────────────────────────────────────────────────────
# Session-state defaults
# ──────────────────────────────────────────────────────────────
def _ss(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


_ss("user", None)
_ss("role", None)
_ss("bucket", "")
_ss("path_prefix", "")
_ss("token_stack", [])
_ss("token", None)
_ss("page_size", 100)
_ss("presign_expiry", 3600)
_ss("filter_text", "")
_ss("filter_mode", "starts with")
_ss("preview_key", None)
_ss("anonymous", False)


def reset_pagination():
    st.session_state.token_stack = []
    st.session_state.token = None
    st.session_state.preview_key = None


def go_to(prefix: str):
    st.session_state.path_prefix = prefix
    st.session_state.filter_text = ""
    reset_pagination()
    st.rerun()


# ──────────────────────────────────────────────────────────────
# Login gate
# ──────────────────────────────────────────────────────────────
if st.session_state.user is None:
    st.title("🗂️ S3 Explorer")
    st.caption("Sign in to browse S3 buckets")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        role = auth.authenticate(username.strip(), password)
        if role:
            st.session_state.user = username.strip()
            st.session_state.role = role
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

role = st.session_state.role
can_write = role in ("admin", "write")


# ──────────────────────────────────────────────────────────────
# S3 client (local credential chain)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_client(anonymous: bool):
    if anonymous:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return boto3.client("s3", config=Config(signature_version="s3v4"))


@st.cache_data(show_spinner=False, ttl=300)
def list_buckets() -> list[str]:
    try:
        return [b["Name"] for b in get_client(False).list_buckets()["Buckets"]]
    except Exception:
        return []


def list_page(bucket: str, prefix: str, page_size: int, token: Optional[str]) -> dict:
    kwargs = dict(Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=page_size)
    if token:
        kwargs["ContinuationToken"] = token
    return get_client(st.session_state.anonymous).list_objects_v2(**kwargs)


def format_bytes(size: Optional[int]) -> str:
    if size is None:
        return "—"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def format_dt(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "—"


def looks_like_text(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    printable = set(string.printable)
    sample = text[:4000]
    return sum(c in printable or c.isprintable() for c in sample) / len(sample) > 0.95


def aws_error(exc: ClientError) -> str:
    err = exc.response.get("Error", {})
    return f"[{err.get('Code', '?')}] {err.get('Message', exc)}"


# ──────────────────────────────────────────────────────────────
# Sidebar — session, bucket, settings
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🗂️ S3 Explorer")
    st.markdown(f"Signed in as **{st.session_state.user}** · role: `{role}`")
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.divider()

    st.subheader("Bucket")
    known = list_buckets()
    options = known + ["Other (type a name)…"]
    if not known:
        st.caption("No buckets listable with local credentials — type a name below.")
    choice = st.selectbox("Accessible buckets", options) if known else "Other (type a name)…"
    if choice == "Other (type a name)…":
        typed = st.text_input(
            "Bucket name",
            placeholder="e.g. a public or cross-account bucket",
        )
        anonymous = st.checkbox(
            "Public bucket (no signing)",
            value=st.session_state.anonymous,
            help="Enable for public buckets when local credentials can't sign for them.",
        )
        selected_bucket = typed.strip()
    else:
        selected_bucket = choice
        anonymous = False

    if st.button("Open bucket", type="primary", use_container_width=True, disabled=not selected_bucket):
        st.session_state.bucket = selected_bucket
        st.session_state.anonymous = anonymous
        st.session_state.path_prefix = ""
        st.session_state.filter_text = ""
        reset_pagination()
        st.rerun()

    st.divider()
    st.subheader("Settings")
    st.session_state.page_size = st.number_input(
        "Page size", min_value=10, max_value=1000, value=st.session_state.page_size, step=10
    )
    st.session_state.presign_expiry = st.number_input(
        "Presigned URL expiry (s)",
        min_value=60,
        max_value=604_800,
        value=st.session_state.presign_expiry,
        step=300,
    )


# ──────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────
tab_names = ["Browser"] + (["User management"] if role == "admin" else [])
tabs = st.tabs(tab_names)


# ══════════════════════════════════════════════════════════════
# Browser tab
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    bucket = st.session_state.bucket
    if not bucket:
        st.info("Pick a bucket in the sidebar and click **Open bucket**.")
    else:
        client = get_client(st.session_state.anonymous)
        path_prefix = st.session_state.path_prefix
        st.markdown(f"#### `s3://{bucket}/{path_prefix}`")

        # Breadcrumbs
        crumbs = [("🪣 root", "")]
        acc = ""
        for part in path_prefix.rstrip("/").split("/"):
            if part:
                acc += part + "/"
                crumbs.append((part, acc))
        crumb_cols = st.columns([1] * len(crumbs) + [max(1, 10 - len(crumbs))])
        for i, (label, prefix) in enumerate(crumbs):
            if crumb_cols[i].button(label, key=f"crumb_{i}"):
                go_to(prefix)

        # Filter + pagination controls
        fcol, mcol, pcol = st.columns([5, 2, 2])
        with fcol:
            filter_text = st.text_input(
                "Filter by filename",
                value=st.session_state.filter_text,
                placeholder="filter by filename…",
                label_visibility="collapsed",
            )
        with mcol:
            filter_mode = st.selectbox(
                "Filter mode",
                ["starts with", "contains"],
                index=["starts with", "contains"].index(st.session_state.filter_mode),
                label_visibility="collapsed",
                help="'starts with' filters server-side (efficient, paginates the whole "
                "bucket). 'contains' filters the current page only.",
            )
        if filter_text != st.session_state.filter_text or filter_mode != st.session_state.filter_mode:
            st.session_state.filter_text = filter_text
            st.session_state.filter_mode = filter_mode
            reset_pagination()
            st.rerun()

        server_prefix = path_prefix + (filter_text if filter_mode == "starts with" else "")

        resp = None
        try:
            resp = list_page(bucket, server_prefix, int(st.session_state.page_size), st.session_state.token)
        except ClientError as exc:
            st.error(f"S3 error: {aws_error(exc)}")
        except Exception as exc:
            st.error(f"Could not list bucket: {exc}")

        if resp is not None:
            next_token = resp.get("NextContinuationToken")
            has_prev = bool(st.session_state.token_stack)
            with pcol:
                b1, b2 = st.columns(2)
                if b1.button("◀ Prev", disabled=not has_prev, use_container_width=True):
                    st.session_state.token = st.session_state.token_stack.pop()
                    st.rerun()
                if b2.button("Next ▶", disabled=not next_token, use_container_width=True):
                    st.session_state.token_stack.append(st.session_state.token)
                    st.session_state.token = next_token
                    st.rerun()

            # Parse rows
            folders = []
            for cp in resp.get("CommonPrefixes", []):
                full = cp["Prefix"]
                folders.append({"name": full[len(path_prefix):], "prefix": full})

            files = []
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") and obj["Size"] == 0:
                    continue  # folder marker
                name = key[len(path_prefix):]
                if filter_mode == "contains" and filter_text and filter_text.lower() not in name.lower():
                    continue
                files.append(
                    {
                        "name": name,
                        "key": key,
                        "size": obj["Size"],
                        "modified": obj["LastModified"].replace(tzinfo=None),
                    }
                )

            # Write-role toolbar
            if can_write:
                with st.expander("✏️ Write actions (upload / new folder)"):
                    up_col, folder_col = st.columns(2)
                    with up_col:
                        uploads = st.file_uploader("Upload files here", accept_multiple_files=True)
                        if uploads and st.button("Upload", type="primary"):
                            for f in uploads:
                                try:
                                    client.upload_fileobj(f, bucket, path_prefix + f.name)
                                    st.success(f"Uploaded `{path_prefix + f.name}`")
                                except ClientError as exc:
                                    st.error(f"Upload of {f.name} failed: {aws_error(exc)}")
                            st.cache_data.clear()
                    with folder_col:
                        new_folder = st.text_input("New folder name")
                        if new_folder and st.button("Create folder"):
                            folder_key = path_prefix + new_folder.strip().strip("/") + "/"
                            try:
                                client.put_object(Bucket=bucket, Key=folder_key)
                                st.success(f"Created `{folder_key}`")
                                st.cache_data.clear()
                            except ClientError as exc:
                                st.error(f"Create folder failed: {aws_error(exc)}")

            st.divider()

            # Listing
            hdr = st.columns([5, 2, 2, 1])
            hdr[0].markdown("**Name**")
            hdr[1].markdown("**Size**")
            hdr[2].markdown("**Modified**")
            hdr[3].markdown("**Actions**")

            if not folders and not files:
                st.info("No objects found here.")

            for row in folders:
                cols = st.columns([5, 2, 2, 1])
                if cols[0].button(f"📁 {row['name']}", key=f"nav_{row['prefix']}"):
                    go_to(row["prefix"])
                cols[1].markdown("—")
                cols[2].markdown("—")

            for row in files:
                cols = st.columns([5, 2, 2, 1])
                cols[0].markdown(f"📄 `{row['name']}`")
                cols[1].markdown(format_bytes(row["size"]))
                cols[2].markdown(format_dt(row["modified"]))
                with cols[3], st.popover("⋯", use_container_width=True):
                    st.caption(f"`{row['key']}`")

                    if st.button("🔗 Presigned URL", key=f"url_{row['key']}"):
                        url = client.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": bucket, "Key": row["key"]},
                            ExpiresIn=int(st.session_state.presign_expiry),
                        )
                        st.code(url, language=None)  # st.code has a built-in copy button
                        st.caption(f"Expires in {st.session_state.presign_expiry}s — copy with the button above.")

                    if st.button("👁 Preview", key=f"prev_{row['key']}"):
                        st.session_state.preview_key = row["key"]
                        st.rerun()

                    try:
                        dl_params = {"Bucket": bucket, "Key": row["key"]}
                        if not st.session_state.anonymous:
                            # S3 rejects response-* params on unsigned (anonymous) requests
                            dl_params["ResponseContentDisposition"] = (
                                f'attachment; filename="{row["name"]}"'
                            )
                        dl_url = client.generate_presigned_url(
                            "get_object",
                            Params=dl_params,
                            ExpiresIn=int(st.session_state.presign_expiry),
                        )
                        st.link_button("⬇ Download", dl_url, use_container_width=True)
                    except Exception as exc:
                        st.error(f"Download link failed: {exc}")

                    if can_write:
                        st.divider()
                        new_name = st.text_input(
                            "Rename / move to key", value=row["key"], key=f"mv_{row['key']}"
                        )
                        if st.button("Move", key=f"mvbtn_{row['key']}") and new_name and new_name != row["key"]:
                            try:
                                client.copy_object(
                                    Bucket=bucket,
                                    Key=new_name,
                                    CopySource={"Bucket": bucket, "Key": row["key"]},
                                )
                                client.delete_object(Bucket=bucket, Key=row["key"])
                                st.success(f"Moved to `{new_name}`")
                                st.cache_data.clear()
                                st.rerun()
                            except ClientError as exc:
                                st.error(aws_error(exc))

                        confirm = st.checkbox("Confirm delete", key=f"cdel_{row['key']}")
                        if st.button("🗑 Delete", key=f"del_{row['key']}", disabled=not confirm):
                            try:
                                client.delete_object(Bucket=bucket, Key=row["key"])
                                st.success(f"Deleted `{row['key']}`")
                                st.cache_data.clear()
                                st.rerun()
                            except ClientError as exc:
                                st.error(aws_error(exc))

            # Preview panel
            if st.session_state.preview_key:
                key = st.session_state.preview_key
                st.divider()
                head_cols = st.columns([8, 1])
                head_cols[0].markdown(f"#### 👁 Preview — `{key}`")
                if head_cols[1].button("✕ Close"):
                    st.session_state.preview_key = None
                    st.rerun()
                try:
                    obj = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{PREVIEW_MAX_BYTES - 1}")
                    data = obj["Body"].read()
                    if looks_like_text(data):
                        truncated = len(data) >= PREVIEW_MAX_BYTES
                        st.code(data.decode("utf-8", errors="replace"), language=None)
                        if truncated:
                            st.caption(f"Showing first {PREVIEW_MAX_BYTES // 1000} KB only.")
                    else:
                        st.warning("This file does not look like ASCII/UTF-8 text — no preview available.")
                except ClientError as exc:
                    st.error(f"Preview failed: {aws_error(exc)}")

            st.divider()
            st.caption(
                f"Page size {st.session_state.page_size} · "
                f"{len(folders)} folders + {len(files)} files on this page"
                + (" · more pages available" if next_token else "")
            )


# ══════════════════════════════════════════════════════════════
# Admin tab — user management
# ══════════════════════════════════════════════════════════════
if role == "admin":
    with tabs[1]:
        st.subheader("Users")
        users = auth.load_users()

        for name, record in sorted(users.items()):
            cols = st.columns([3, 2, 3, 2, 2])
            cols[0].markdown(f"**{name}**" + (" (you)" if name == st.session_state.user else ""))
            new_role = cols[1].selectbox(
                "Role", auth.ROLES, index=auth.ROLES.index(record["role"]),
                key=f"role_{name}", label_visibility="collapsed",
            )
            if new_role != record["role"]:
                try:
                    auth.set_role(name, new_role)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            new_pw = cols[2].text_input(
                "New password", key=f"pw_{name}", type="password",
                placeholder="new password", label_visibility="collapsed",
            )
            if cols[3].button("Set password", key=f"setpw_{name}", disabled=not new_pw):
                auth.set_password(name, new_pw)
                st.success(f"Password updated for {name}")
            if cols[4].button("Delete", key=f"deluser_{name}", disabled=name == st.session_state.user):
                try:
                    auth.delete_user(name)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

        st.divider()
        st.subheader("Add user")
        with st.form("add_user", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            u = c1.text_input("Username")
            p = c2.text_input("Password", type="password")
            r = c3.selectbox("Role", auth.ROLES, index=auth.ROLES.index("read-only"))
            if st.form_submit_button("Create user", type="primary"):
                if not u.strip() or not p:
                    st.error("Username and password are required.")
                else:
                    try:
                        auth.add_user(u.strip(), p, r)
                        st.success(f"Created user '{u.strip()}' with role {r}")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
