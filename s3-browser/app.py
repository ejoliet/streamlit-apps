"""
S3 Private Bucket Browser
Replicates the qoomon S3 Bucket Browser HTML tool for private buckets
using boto3 + IAM credentials (presigned URLs for downloads).
"""

import io
import logging
import zipfile
from datetime import datetime
from typing import Optional

import boto3
import streamlit as st
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="S3 Bucket Browser",
    page_icon="🪣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Session-state defaults
# ──────────────────────────────────────────────────────────────
def _ss(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


_ss("path_prefix", "")
_ss("continuation_tokens", [])   # stack of previous tokens
_ss("continuation_token", None)  # current page token
_ss("search_prefix", "")
_ss("sort_col", "name")
_ss("sort_asc", True)
_ss("s3_client", None)
_ss("bucket", "")
_ss("root_prefix", "")
_ss("page_size", 50)
_ss("presign_expiry", 3600)
_ss("exclude_index_html", True)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def format_bytes(size: Optional[int]) -> str:
    if not size:
        return "—"
    if size < 1024:
        return f"{size} B"
    if size < 1_048_576:
        return f"{size / 1024:.0f} KB"
    if size < 1_073_741_824:
        return f"{size / 1_048_576:.2f} MB"
    return f"{size / 1_073_741_824:.2f} GB"


def format_dt(dt: Optional[datetime]) -> str:
    if not dt:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def bucket_prefix() -> str:
    root = st.session_state.root_prefix
    path = st.session_state.path_prefix
    return f"{root}{path}"


def build_s3_client(
    auth_method: str,
    profile: str,
    access_key: str,
    secret_key: str,
    session_token: str,
    region: str,
) -> boto3.client:
    """Build a boto3 S3 client from sidebar credentials."""
    cfg = Config(signature_version="s3v4", region_name=region or None)
    if auth_method == "AWS Profile":
        session = boto3.Session(profile_name=profile or None)
    elif auth_method == "Access Key / Secret":
        session = boto3.Session(
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
            aws_session_token=session_token or None,
            region_name=region or None,
        )
    else:  # Instance / environment credentials
        session = boto3.Session(region_name=region or None)
    return session.client("s3", config=cfg)


def list_objects(
    client,
    bucket: str,
    prefix: str,
    page_size: int,
    continuation_token: Optional[str],
) -> dict:
    """Call list_objects_v2 and return the raw response."""
    kwargs = dict(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter="/",
        MaxKeys=page_size,
    )
    if continuation_token:
        kwargs["ContinuationToken"] = continuation_token
    return client.list_objects_v2(**kwargs)


def presigned_url(client, bucket: str, key: str, expiry: int) -> str:
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry,
    )


def download_object_bytes(client, bucket: str, key: str) -> bytes:
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


# ──────────────────────────────────────────────────────────────
# Sidebar — credentials & config
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🪣 S3 Bucket Browser")
    st.caption("Private bucket edition — powered by boto3")
    st.divider()

    st.subheader("AWS Credentials")
    auth_method = st.selectbox(
        "Auth method",
        ["Instance / Environment", "AWS Profile", "Access Key / Secret"],
        help=(
            "Instance/Environment: uses EC2 instance role, ECS task role, or "
            "AWS_ACCESS_KEY_ID env vars.\n\n"
            "AWS Profile: reads from ~/.aws/credentials.\n\n"
            "Access Key / Secret: enter keys below."
        ),
    )

    profile_name = ""
    access_key = secret_key = session_token = ""
    region = ""

    if auth_method == "AWS Profile":
        profile_name = st.text_input("Profile name", value="default")
        region = st.text_input("Region (optional)", placeholder="us-east-1")
    elif auth_method == "Access Key / Secret":
        access_key = st.text_input("Access Key ID", type="password")
        secret_key = st.text_input("Secret Access Key", type="password")
        session_token = st.text_input("Session Token (optional)", type="password")
        region = st.text_input("Region", placeholder="us-east-1")
    else:
        region = st.text_input("Region (optional)", placeholder="us-east-1")

    st.divider()
    st.subheader("Bucket Config")
    bucket_input = st.text_input("Bucket name", value=st.session_state.bucket)
    root_prefix_input = st.text_input(
        "Root prefix",
        value=st.session_state.root_prefix,
        placeholder="subfolder/  (optional)",
        help="Limit the browser to a sub-path within the bucket.",
    )
    page_size_input = st.number_input(
        "Page size", min_value=1, max_value=1000, value=st.session_state.page_size, step=10
    )
    presign_expiry_input = st.number_input(
        "Presigned URL expiry (s)", min_value=60, max_value=604800,
        value=st.session_state.presign_expiry, step=300
    )
    exclude_index = st.checkbox("Exclude index.html", value=st.session_state.exclude_index_html)

    connect_btn = st.button("Connect", type="primary", use_container_width=True)

    if connect_btn:
        try:
            client = build_s3_client(
                auth_method, profile_name, access_key, secret_key, session_token, region
            )
            # Quick connectivity check
            client.head_bucket(Bucket=bucket_input)

            # Persist to session state
            st.session_state.s3_client = client
            st.session_state.bucket = bucket_input
            root = root_prefix_input.strip("/")
            st.session_state.root_prefix = f"{root}/" if root else ""
            st.session_state.page_size = int(page_size_input)
            st.session_state.presign_expiry = int(presign_expiry_input)
            st.session_state.exclude_index_html = exclude_index

            # Reset navigation
            st.session_state.path_prefix = ""
            st.session_state.continuation_tokens = []
            st.session_state.continuation_token = None
            st.session_state.search_prefix = ""

            st.success(f"Connected to **{bucket_input}**")
        except ProfileNotFound as exc:
            st.error(f"AWS profile not found: {exc}")
        except NoCredentialsError:
            st.error("No AWS credentials found.")
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            st.error(f"AWS error [{code}]: {exc.response['Error']['Message']}")
        except Exception as exc:
            st.error(f"Connection failed: {exc}")


# ──────────────────────────────────────────────────────────────
# Main — guard: not connected
# ──────────────────────────────────────────────────────────────
client = st.session_state.s3_client
bucket = st.session_state.bucket

if not client or not bucket:
    st.info("Configure credentials and bucket in the sidebar, then click **Connect**.")
    st.stop()


# ──────────────────────────────────────────────────────────────
# Main — fetch current page
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=30)
def _cached_list(
    bucket: str,
    prefix: str,
    page_size: int,
    continuation_token: Optional[str],
    # cache-bust key: auth identity changes break the cache naturally via different clients
    _cache_key: str = "",
) -> dict:
    return list_objects(
        st.session_state.s3_client, bucket, prefix, page_size, continuation_token
    )


try:
    response = _cached_list(
        bucket,
        bucket_prefix(),
        st.session_state.page_size,
        st.session_state.continuation_token,
        _cache_key=f"{bucket}/{bucket_prefix()}",
    )
except ClientError as exc:
    st.error(f"S3 error: {exc.response['Error']['Message']}")
    st.stop()
except Exception as exc:
    st.error(f"Unexpected error: {exc}")
    st.stop()


# ──────────────────────────────────────────────────────────────
# Parse response → rows
# ──────────────────────────────────────────────────────────────
root_prefix = st.session_state.root_prefix
exclude_index = st.session_state.exclude_index_html
path_prefix = st.session_state.path_prefix

folders = []
for cp in response.get("CommonPrefixes", []):
    raw_prefix = cp["Prefix"]
    rel_prefix = raw_prefix[len(root_prefix):]  # strip root
    name = rel_prefix.rstrip("/").split("/")[-1] + "/"
    folders.append({"type": "prefix", "name": name, "prefix": rel_prefix, "size": None, "modified": None})

files = []
for obj in response.get("Contents", []):
    key = obj["Key"]
    rel_key = key[len(root_prefix):]
    # skip directory markers and index.html if requested
    if key == bucket_prefix():
        continue
    if exclude_index and rel_key == "index.html":
        continue
    if key.endswith("/") and obj["Size"] == 0:
        continue
    name = key.split("/")[-1]
    files.append({
        "type": "content",
        "name": name,
        "key": key,
        "rel_key": rel_key,
        "size": obj["Size"],
        "modified": obj["LastModified"].replace(tzinfo=None),
    })

next_token = response.get("NextContinuationToken")


# ──────────────────────────────────────────────────────────────
# Sort
# ──────────────────────────────────────────────────────────────
col_map = {"name": "name", "size": "size", "modified": "modified"}


def sort_rows(rows: list[dict]) -> list[dict]:
    col = st.session_state.sort_col
    asc = st.session_state.sort_asc
    return sorted(
        rows,
        key=lambda r: (r[col] is None, r[col] if r[col] is not None else ""),
        reverse=not asc,
    )


folders_sorted = sort_rows(folders)
files_sorted = sort_rows(files)
all_rows = folders_sorted + files_sorted  # folders always first


# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown("## 🪣 S3 Bucket Browser")
st.caption(f"Bucket: **{bucket}**  ·  Prefix: `{bucket_prefix() or '/'}`")
st.divider()


# ──────────────────────────────────────────────────────────────
# Breadcrumbs
# ──────────────────────────────────────────────────────────────
parts = [""] + [p + "/" for p in path_prefix.rstrip("/").split("/") if p]
breadcrumb_cols = st.columns(len(parts), gap="small")

for i, part in enumerate(parts):
    accumulated = "".join(parts[1 : i + 1])
    label = "🪣 root" if i == 0 else part
    with breadcrumb_cols[i]:
        if st.button(label, key=f"bc_{i}_{accumulated}", use_container_width=False):
            st.session_state.path_prefix = accumulated
            st.session_state.continuation_tokens = []
            st.session_state.continuation_token = None
            st.session_state.search_prefix = accumulated.rstrip("/").split("/")[-1] + "/" if accumulated else ""
            st.rerun()


# ──────────────────────────────────────────────────────────────
# Search + pagination controls
# ──────────────────────────────────────────────────────────────
ctrl_left, ctrl_right = st.columns([4, 1])

with ctrl_left:
    search_val = st.text_input(
        "Search prefix",
        value=st.session_state.search_prefix,
        placeholder="filter by prefix…",
        label_visibility="collapsed",
    )
    if search_val != st.session_state.search_prefix:
        # Navigate to new prefix
        base_dir = path_prefix.rsplit("/", 1)[0] + "/" if "/" in path_prefix else ""
        st.session_state.path_prefix = base_dir + search_val
        st.session_state.search_prefix = search_val
        st.session_state.continuation_tokens = []
        st.session_state.continuation_token = None
        st.rerun()

has_prev = len(st.session_state.continuation_tokens) > 0
has_next = bool(next_token)

with ctrl_right:
    pg_cols = st.columns(2)
    with pg_cols[0]:
        if st.button("◀", disabled=not has_prev, use_container_width=True):
            st.session_state.continuation_token = st.session_state.continuation_tokens.pop()
            st.rerun()
    with pg_cols[1]:
        if st.button("▶", disabled=not has_next, use_container_width=True):
            st.session_state.continuation_tokens.append(st.session_state.continuation_token)
            st.session_state.continuation_token = next_token
            st.rerun()


# ──────────────────────────────────────────────────────────────
# Download-all as ZIP
# ──────────────────────────────────────────────────────────────
if len(files_sorted) >= 2:
    if st.button(f"⬇ Download all {len(files_sorted)} files as ZIP"):
        zip_buffer = io.BytesIO()
        progress = st.progress(0, text="Building ZIP…")
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, row in enumerate(files_sorted):
                try:
                    data = download_object_bytes(client, bucket, row["key"])
                    zf.writestr(row["name"], data)
                except ClientError as exc:
                    st.warning(f"Skipped {row['name']}: {exc.response['Error']['Message']}")
                progress.progress((idx + 1) / len(files_sorted), text=f"{idx + 1}/{len(files_sorted)} files…")
        progress.empty()
        zip_buffer.seek(0)
        folder_name = path_prefix.rstrip("/").split("/")[-1] or bucket
        st.download_button(
            label="💾 Save ZIP",
            data=zip_buffer,
            file_name=f"{folder_name}.zip",
            mime="application/zip",
        )

st.divider()


# ──────────────────────────────────────────────────────────────
# Sort header row
# ──────────────────────────────────────────────────────────────
def sort_btn(label: str, col: str) -> None:
    arrow = ""
    if st.session_state.sort_col == col:
        arrow = " ▲" if st.session_state.sort_asc else " ▼"
    if st.button(f"{label}{arrow}", key=f"sort_{col}", use_container_width=True):
        if st.session_state.sort_col == col:
            st.session_state.sort_asc = not st.session_state.sort_asc
        else:
            st.session_state.sort_col = col
            st.session_state.sort_asc = True
        st.rerun()


hdr = st.columns([5, 2, 2, 1])
with hdr[0]:
    sort_btn("Name", "name")
with hdr[1]:
    sort_btn("Size", "size")
with hdr[2]:
    sort_btn("Modified", "modified")
with hdr[3]:
    st.markdown("&nbsp;", unsafe_allow_html=True)  # actions column header placeholder

st.divider()


# ──────────────────────────────────────────────────────────────
# File / folder rows
# ──────────────────────────────────────────────────────────────
if not all_rows:
    st.info("No objects found at this prefix.")

for row in all_rows:
    row_cols = st.columns([5, 2, 2, 1])

    if row["type"] == "prefix":
        # Folder
        with row_cols[0]:
            if st.button(f"📁  {row['name']}", key=f"nav_{row['prefix']}", use_container_width=False):
                st.session_state.path_prefix = row["prefix"]
                st.session_state.continuation_tokens = []
                st.session_state.continuation_token = None
                st.session_state.search_prefix = row["name"]
                st.rerun()
        with row_cols[1]:
            st.markdown("—")
        with row_cols[2]:
            st.markdown("—")
        with row_cols[3]:
            st.markdown("")
    else:
        # File
        with row_cols[0]:
            st.markdown(f"📄  `{row['name']}`")
        with row_cols[1]:
            st.markdown(format_bytes(row["size"]))
        with row_cols[2]:
            st.markdown(format_dt(row["modified"]))
        with row_cols[3]:
            try:
                url = presigned_url(client, bucket, row["key"], st.session_state.presign_expiry)
                st.link_button("⬇", url, use_container_width=True)
            except ClientError as exc:
                st.error(exc.response["Error"]["Message"])


# ──────────────────────────────────────────────────────────────
# Bottom pagination
# ──────────────────────────────────────────────────────────────
if has_prev or has_next:
    st.divider()
    bp_cols = st.columns([1, 1, 8])
    with bp_cols[0]:
        if st.button("◀ Prev", disabled=not has_prev, use_container_width=True):
            st.session_state.continuation_token = st.session_state.continuation_tokens.pop()
            st.rerun()
    with bp_cols[1]:
        if st.button("Next ▶", disabled=not has_next, use_container_width=True):
            st.session_state.continuation_tokens.append(st.session_state.continuation_token)
            st.session_state.continuation_token = next_token
            st.rerun()


# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Bucket: `s3://{bucket}/{bucket_prefix()}`  ·  "
    f"Page size: {st.session_state.page_size}  ·  "
    f"Presigned URL expiry: {st.session_state.presign_expiry}s"
)
