import base64
import os
import secrets
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests
import streamlit as st


CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8501")
API_BASE = "https://api.github.com"


def _build_headers(token: str, accept_preview: bool = False) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    if accept_preview:
        headers["Accept"] = "application/vnd.github.mercy-preview+json"
    return headers


def _get_query_params() -> Dict[str, List[str]]:
    params_accessor = getattr(st, "query_params", None)
    if params_accessor is not None:
        return {
            key: value if isinstance(value, list) else [value]
            for key, value in dict(params_accessor).items()
        }
    legacy_getter = getattr(st, "experimental_get_query_params", None)
    if legacy_getter:
        return legacy_getter()
    return {}


def _clear_query_params() -> None:
    params_accessor = getattr(st, "query_params", None)
    if params_accessor is not None:
        params_accessor.clear()
        return
    legacy_setter = getattr(st, "experimental_set_query_params", None)
    if legacy_setter:
        legacy_setter()


def _exchange_code_for_token(code: str) -> Optional[str]:
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "redirect_uri": REDIRECT_URI,
    }
    resp = requests.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data=payload,
        timeout=10,
    )
    if resp.status_code != 200:
        st.error("Failed to exchange OAuth code for an access token.")
        return None
    token = resp.json().get("access_token")
    if not token:
        st.error("GitHub did not return an access token.")
    return token


@st.cache_data(show_spinner=False)
def _fetch_repositories(token: str) -> List[Dict]:
    repos: List[Dict] = []
    page = 1
    headers = _build_headers(token, accept_preview=True)
    while True:
        params = {
            "per_page": 100,
            "page": page,
            "sort": "updated",
            "direction": "desc",
        }
        resp = requests.get(
            f"{API_BASE}/user/repos", headers=headers, params=params, timeout=10
        )
        resp.raise_for_status()
        page_data = resp.json()
        if not page_data:
            break
        for repo in page_data:
            topics = repo.get("topics", [])
            if repo["name"].startswith("roman-") and "roman-docs" in topics:
                repos.append(repo)
        if len(page_data) < 100:
            break
        page += 1
    return repos


@st.cache_data(show_spinner=False)
def _fetch_activity(token: str, full_name: str) -> Dict[str, str]:
    headers = _build_headers(token)
    release_resp = requests.get(
        f"{API_BASE}/repos/{full_name}/releases/latest", headers=headers, timeout=10
    )
    if release_resp.status_code == 200:
        release = release_resp.json()
        return {
            "label": f"Release {release.get('tag_name', 'latest')}",
            "timestamp": release.get("published_at", "unknown date"),
            "details": release.get("name") or release.get("body") or "",
        }

    commits_resp = requests.get(
        f"{API_BASE}/repos/{full_name}/commits",
        headers=headers,
        params={"per_page": 1},
        timeout=10,
    )
    commits_resp.raise_for_status()
    commits = commits_resp.json()
    if commits:
        commit = commits[0]
        message = commit["commit"]["message"].splitlines()[0]
        sha = commit["sha"][:7]
        timestamp = commit["commit"]["committer"]["date"]
        return {"label": f"Commit {sha}", "timestamp": timestamp, "details": message}
    return {"label": "No activity", "timestamp": "", "details": ""}


@st.cache_data(show_spinner=False)
def _fetch_primary_markdown(token: str, full_name: str, ref: str) -> Optional[str]:
    headers = _build_headers(token)
    readme_resp = requests.get(
        f"{API_BASE}/repos/{full_name}/readme",
        headers=headers,
        params={"ref": ref},
        timeout=10,
    )
    if readme_resp.status_code == 200:
        data = readme_resp.json()
        return _decode_markdown(data.get("content"), data.get("encoding"))

    contents_resp = requests.get(
        f"{API_BASE}/repos/{full_name}/contents",
        headers=headers,
        params={"ref": ref},
        timeout=10,
    )
    if contents_resp.status_code != 200:
        return None
    for item in contents_resp.json():
        if item["name"].lower().endswith(".md"):
            md_resp = requests.get(
                f"{API_BASE}/repos/{full_name}/contents/{item['path']}",
                headers=headers,
                params={"ref": ref},
                timeout=10,
            )
            if md_resp.status_code == 200:
                data = md_resp.json()
                return _decode_markdown(data.get("content"), data.get("encoding"))
    return None


def _decode_markdown(content: Optional[str], encoding: Optional[str]) -> Optional[str]:
    if not content:
        return None
    if encoding == "base64":
        return base64.b64decode(content).decode("utf-8", errors="ignore")
    return content


def _handle_oauth_callback() -> None:
    params = _get_query_params()
    if "code" not in params:
        return

    code = params["code"][0]
    incoming_state = params.get("state", [""])[0]
    saved_state = st.session_state.get("oauth_state")
    if not saved_state or incoming_state != saved_state:
        st.error("OAuth state mismatch. Please try logging in again.")
        return

    token = _exchange_code_for_token(code)
    if token:
        st.session_state["access_token"] = token
        st.session_state["oauth_state"] = None
        _clear_query_params()


def _login_view() -> None:
    if not CLIENT_ID or not CLIENT_SECRET:
        st.warning(
            "Configure GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET environment variables "
            "to enable GitHub authentication."
        )
        st.stop()

    if "oauth_state" not in st.session_state or not st.session_state["oauth_state"]:
        st.session_state["oauth_state"] = secrets.token_urlsafe(32)

    query = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": "repo",
        "state": st.session_state["oauth_state"],
        "allow_signup": "false",
    }
    authorize_url = f"https://github.com/login/oauth/authorize?{urlencode(query)}"
    st.info(
        "Authenticate with GitHub to view docs repositories. "
        "If you are running this locally, ensure your OAuth app redirect URI "
        f"is set to {REDIRECT_URI}."
    )
    st.link_button("Log In with GitHub", authorize_url, use_container_width=True)


def _render_portal(token: str) -> None:
    repos = _fetch_repositories(token)
    if not repos:
        st.info("No repositories match the roman-* + roman-docs criteria.")
        return

    repo_entries = []
    for repo in repos:
        activity = _fetch_activity(token, repo["full_name"])
        repo_entries.append({"repo": repo, "activity": activity})

    repo_options = [
        f"{entry['repo']['name']} · {entry['activity']['label']}"
        for entry in repo_entries
    ]

    names = [entry["repo"]["name"] for entry in repo_entries]
    default_index = 0
    selected_name = st.session_state.get("selected_repo")
    if selected_name in names:
        default_index = names.index(selected_name)

    left, right = st.columns((1, 2), gap="large")
    with left:
        st.subheader("Repositories")
        choice = st.radio(
            "Select a repository",
            options=range(len(repo_entries)),
            format_func=lambda idx: repo_options[idx],
            index=default_index,
            label_visibility="collapsed",
        )
        selected_entry = repo_entries[choice]
        st.session_state["selected_repo"] = selected_entry["repo"]["name"]

        activity = selected_entry["activity"]
        st.caption(
            f"Latest activity: {activity['label']} — {activity.get('timestamp', '')}"
        )
    with right:
        repo = selected_entry["repo"]
        st.subheader(repo["name"])
        st.markdown(f"[Open on GitHub]({repo['html_url']})")
        st.caption(repo.get("description") or "No description provided.")

        markdown = _fetch_primary_markdown(
            token, repo["full_name"], repo["default_branch"]
        )
        if markdown:
            st.markdown(markdown, unsafe_allow_html=False)
        else:
            st.warning("No markdown content found in the repository root.")


def main() -> None:
    st.set_page_config(page_title="Roman Docs Portal", layout="wide")
    st.title("Roman Docs Access Portal")

    _handle_oauth_callback()

    token = st.session_state.get("access_token")
    if not token:
        _login_view()
        return

    _render_portal(token)


if __name__ == "__main__":
    main()
