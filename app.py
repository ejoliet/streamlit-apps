# pip install streamlit requests

import base64
import os
import re
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:
    st = None

GITHUB_API = "https://api.github.com"
DEFAULT_DOC_TOPIC = "roman-docs"
REPO_DOC_PATTERN = re.compile(r"^roman-ssc_d-([mtd])(\d{3})$", re.IGNORECASE)


class GitHubAPIError(RuntimeError):
    """Raised when the GitHub API returns an error response."""


def streamlit_available() -> bool:
    return st is not None


def normalize_path(path: str) -> str:
    return path.strip().strip("/")


def get_token(
    session_state: Optional[Dict[str, Any]] = None,
    query_params: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    session_state = session_state or {}
    query_params = query_params or {}
    env = env or os.environ

    session_token = session_state.get("github_token")
    if session_token:
        return str(session_token)

    query_token = query_params.get("token")
    if isinstance(query_token, list):
        query_token = query_token[0] if query_token else None
    if query_token:
        return str(query_token)

    env_token = env.get("GITHUB_TOKEN")
    if env_token:
        return env_token

    return None


def gh_get(path: str, token: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    response = requests.get(f"{GITHUB_API}{path}", headers=headers, params=params, timeout=20)
    return response


def ensure_ok(response: requests.Response, context: str) -> None:
    if response.ok:
        return

    detail = ""
    try:
        payload = response.json()
        detail = payload.get("message", "") if isinstance(payload, dict) else str(payload)
    except Exception:
        detail = response.text[:300]

    raise GitHubAPIError(f"{context} failed ({response.status_code}): {detail}")


def fetch_user(token: str) -> Dict[str, Any]:
    r = gh_get("/user", token)
    ensure_ok(r, "Fetch user")
    data = r.json()
    return data if isinstance(data, dict) else {}


def fetch_repos(token: str, affiliation: str = "owner,collaborator,organization_member") -> List[Dict[str, Any]]:
    repos: List[Dict[str, Any]] = []
    page = 1
    while True:
        r = gh_get(
            "/user/repos",
            token,
            params={
                "per_page": 100,
                "page": page,
                "sort": "updated",
                "direction": "desc",
                "affiliation": affiliation,
            },
        )
        ensure_ok(r, "Fetch repositories")
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        repos.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return repos


def fetch_branches(token: str, owner: str, repo: str) -> List[Dict[str, Any]]:
    r = gh_get(f"/repos/{owner}/{repo}/branches", token, params={"per_page": 100})
    ensure_ok(r, "Fetch branches")
    data = r.json()
    return data if isinstance(data, list) else []


def fetch_tags(token: str, owner: str, repo: str) -> List[Dict[str, Any]]:
    r = gh_get(f"/repos/{owner}/{repo}/tags", token, params={"per_page": 100})
    ensure_ok(r, "Fetch tags")
    data = r.json()
    return data if isinstance(data, list) else []


def fetch_contents(token: str, owner: str, repo: str, path: str = "", ref: Optional[str] = None) -> List[Dict[str, Any]]:
    clean_path = normalize_path(path)
    params = {"ref": ref} if ref else None
    endpoint = f"/repos/{owner}/{repo}/contents/{clean_path}" if clean_path else f"/repos/{owner}/{repo}/contents"
    r = gh_get(endpoint, token, params=params)
    ensure_ok(r, "Fetch contents")
    data = r.json()
    return data if isinstance(data, list) else [data]


def fetch_file_content(token: str, owner: str, repo: str, path: str, ref: Optional[str] = None) -> bytes:
    params = {"ref": ref} if ref else None
    r = gh_get(f"/repos/{owner}/{repo}/contents/{normalize_path(path)}", token, params=params)
    ensure_ok(r, f"Fetch file {path}")
    payload = r.json()
    if not isinstance(payload, dict):
        raise GitHubAPIError(f"Fetch file {path} failed: unexpected response")

    encoded = payload.get("content")
    encoding = payload.get("encoding")
    download_url = payload.get("download_url")

    if encoded and encoding == "base64":
        return base64.b64decode(encoded)
    if download_url:
        raw = requests.get(download_url, timeout=20)
        if raw.ok:
            return raw.content

    raise GitHubAPIError(f"Fetch file {path} failed: file content unavailable")


def repo_matches(repo: Dict[str, Any], text: str) -> bool:
    haystack = " ".join(
        [
            repo.get("full_name", ""),
            repo.get("name", ""),
            repo.get("description") or "",
            repo.get("language") or "",
            " ".join(topic for topic in repo.get("topics", [])),
        ]
    ).lower()
    return text.lower() in haystack


def filter_repos(
    repos: List[Dict[str, Any]],
    query: str,
    repo_visibility: str,
    archived_mode: str,
    required_topic: Optional[str] = None,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for repo in repos:
        topics = [str(topic).lower() for topic in repo.get("topics", [])]
        if required_topic and required_topic.lower() not in topics:
            continue
        if query and not repo_matches(repo, query):
            continue
        if repo_visibility != "all" and repo.get("private", False) != (repo_visibility == "private"):
            continue
        if archived_mode == "exclude" and repo.get("archived"):
            continue
        if archived_mode == "only" and not repo.get("archived"):
            continue
        filtered.append(repo)
    return filtered


def infer_main_document_name(repo_name: str) -> List[str]:
    match = REPO_DOC_PATTERN.match(repo_name)
    if not match:
        return []
    letter, number = match.groups()
    stem = f"SSC_D-{letter.upper()}{number}"
    return [f"{stem}.md", f"{stem}.pdf", f"{stem}.docx"]


def flatten_files(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in contents if item.get("type") == "file" and item.get("name")]


def choose_main_document(repo_name: str, files: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    by_name = {str(item.get("name")): item for item in files}
    for candidate in infer_main_document_name(repo_name):
        if candidate in by_name:
            return by_name[candidate]

    md_candidates = sorted(
        [item for item in files if str(item.get("name", "")).lower().endswith(".md")],
        key=lambda item: str(item.get("name", "")).lower(),
    )
    if md_candidates:
        return md_candidates[0]

    pdf_candidates = sorted(
        [item for item in files if str(item.get("name", "")).lower().endswith(".pdf")],
        key=lambda item: str(item.get("name", "")).lower(),
    )
    if pdf_candidates:
        return pdf_candidates[0]

    docx_candidates = sorted(
        [item for item in files if str(item.get("name", "")).lower().endswith(".docx")],
        key=lambda item: str(item.get("name", "")).lower(),
    )
    if docx_candidates:
        return docx_candidates[0]

    return None


def version_sort_key(name: str) -> Tuple[int, int, str]:
    lowered = name.lower()
    if lowered == "main":
        return (0, 0, lowered)
    numeric_match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", lowered)
    if numeric_match:
        major = int(numeric_match.group(1) or 0)
        minor = int(numeric_match.group(2) or 0)
        patch = int(numeric_match.group(3) or 0)
        return (1, major * 1000000 + minor * 1000 + patch, lowered)
    return (2, 0, lowered)


def build_version_options(default_branch: str, branches: List[Dict[str, Any]], tags: List[Dict[str, Any]]) -> List[str]:
    options = []
    seen = set()

    preferred_main = default_branch or "main"
    if preferred_main not in seen:
        options.append(preferred_main)
        seen.add(preferred_main)

    branch_names = sorted([b.get("name", "") for b in branches if b.get("name")], key=version_sort_key, reverse=True)
    tag_names = sorted([t.get("name", "") for t in tags if t.get("name")], key=version_sort_key, reverse=True)

    for name in tag_names + branch_names:
        if name and name not in seen:
            options.append(name)
            seen.add(name)

    return options


def select_default_version(default_branch: str, options: List[str], tags: List[Dict[str, Any]]) -> str:
    if tags:
        sorted_tags = sorted([t.get("name", "") for t in tags if t.get("name")], key=version_sort_key, reverse=True)
        if sorted_tags:
            return sorted_tags[0]
    if default_branch and default_branch in options:
        return default_branch
    if "main" in options:
        return "main"
    return options[0] if options else "main"


def render_document(token: str, owner: str, repo_name: str, ref: str, doc_item: Dict[str, Any]) -> None:
    if st is None:
        return

    doc_name = str(doc_item.get("name", ""))
    doc_path = str(doc_item.get("path", doc_name))
    lowered = doc_name.lower()

    try:
        content = fetch_file_content(token, owner, repo_name, doc_path, ref=ref)
    except GitHubAPIError as e:
        st.error(str(e))
        return

    st.subheader(doc_name)
    st.caption(f"Reference: {ref}")

    if lowered.endswith(".md"):
        try:
            st.markdown(content.decode("utf-8"))
        except UnicodeDecodeError:
            st.code(content[:5000])
    elif lowered.endswith(".pdf"):
        st.download_button(
            "Download PDF",
            data=content,
            file_name=doc_name,
            mime="application/pdf",
        )
        pdf_b64 = base64.b64encode(content).decode("utf-8")
        st.components.v1.iframe(f"data:application/pdf;base64,{pdf_b64}", height=900, scrolling=True)
    elif lowered.endswith(".docx"):
        st.download_button(
            "Download DOCX",
            data=content,
            file_name=doc_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        st.info("DOCX preview is not rendered inline. Download the file to inspect it.")
    else:
        st.download_button("Download file", data=content, file_name=doc_name, mime="application/octet-stream")



def render_missing_streamlit_message() -> None:
    message = (
        "This file defines a Streamlit app, but Streamlit is not installed in the current Python environment.\n\n"
        "Install dependencies and run the app with:\n"
        "  pip install streamlit requests\n"
        "  streamlit run streamlit_github_repo_navigator_app.py\n\n"
        "The module no longer crashes on import without Streamlit, which makes it testable in constrained environments."
    )
    print(message)


def run_streamlit_app() -> None:
    if st is None:
        render_missing_streamlit_message()
        return

    st.set_page_config(page_title="Roman Documentation Navigator", layout="wide")
    st.title("Roman Documentation Navigator")
    st.caption("Documentation-first GitHub browser with default filtering on the roman-docs topic.")

    with st.sidebar:
        st.header("Authentication")
        st.warning(
            "This app cannot directly reuse GitHub credentials already stored for github.com in your browser. "
            "Use a token passed to the app via sidebar input, environment variable, or URL query parameter."
        )
        token_input = st.text_input("GitHub token", type="password", help="Needs repo access for private repositories.")
        if token_input:
            st.session_state["github_token"] = token_input.strip()
        if st.button("Clear token"):
            st.session_state.pop("github_token", None)
            st.query_params.clear()
            st.rerun()

        st.markdown("### Token sources")
        st.code(
            "1. Sidebar input\n2. URL: ?token=ghp_xxx\n3. Env: GITHUB_TOKEN",
            language="text",
        )

    token = get_token(session_state=st.session_state, query_params=st.query_params)
    if not token:
        st.info("Provide a GitHub token to continue.")
        st.stop()

    try:
        user = fetch_user(token)
    except GitHubAPIError as e:
        st.error(str(e))
        st.stop()

    st.success(f"Authenticated as {user.get('login')}")

    with st.sidebar:
        st.header("Repository filters")
        only_docs = st.checkbox("Only documentation repos", value=True)
        query = st.text_input("Search repositories", value="")
        repo_visibility = st.selectbox("Visibility", ["all", "public", "private"], index=0)
        archived_mode = st.selectbox("Archived", ["exclude", "only", "include"], index=0)
        affiliation = st.multiselect(
            "Affiliation",
            ["owner", "collaborator", "organization_member"],
            default=["owner", "collaborator", "organization_member"],
        )

    try:
        repos = fetch_repos(token, affiliation=",".join(affiliation))
    except GitHubAPIError as e:
        st.error(str(e))
        st.stop()

    filtered = filter_repos(
        repos,
        query=query,
        repo_visibility=repo_visibility,
        archived_mode=archived_mode,
        required_topic=DEFAULT_DOC_TOPIC if only_docs else None,
    )

    repo_options = {repo["full_name"]: repo for repo in filtered if "full_name" in repo}
    repo_names = list(repo_options.keys())

    if not repo_names:
        st.warning("No repositories matched the current filters.")
        st.stop()

    st.subheader("Repository selection")
    selected_repo_name = st.selectbox("Documentation repository", repo_names)
    repo = repo_options[selected_repo_name]
    owner = repo["owner"]["login"]
    repo_name = repo["name"]
    default_branch = str(repo.get("default_branch") or "main")

    try:
        root_contents = fetch_contents(token, owner, repo_name, "", ref=default_branch)
        root_files = flatten_files(root_contents)
    except GitHubAPIError as e:
        st.error(str(e))
        st.stop()

    try:
        tags = fetch_tags(token, owner, repo_name)
    except GitHubAPIError:
        tags = []

    try:
        branches = fetch_branches(token, owner, repo_name)
    except GitHubAPIError:
        branches = []

    version_options = build_version_options(default_branch, branches, tags)
    default_version = select_default_version(default_branch, version_options, tags)
    default_index = version_options.index(default_version) if default_version in version_options else 0

    info_col, control_col = st.columns([1, 2])
    with info_col:
        st.markdown(f"**Repository:** {repo.get('full_name')}  ")
        st.markdown(f"**Description:** {repo.get('description') or '—'}  ")
        st.markdown(f"**Topics:** {', '.join(repo.get('topics', [])) or '—'}  ")
        st.markdown(f"**Default branch:** {default_branch}  ")
        st.link_button("Open repository on GitHub", repo.get("html_url", "https://github.com"))

    with control_col:
        selected_version = st.selectbox("Version / release", version_options, index=default_index)
        tag_names = [tag.get("name", "") for tag in tags if tag.get("name")]
        st.multiselect("Available tags", tag_names, default=tag_names, disabled=True)

    try:
        current_contents = fetch_contents(token, owner, repo_name, "", ref=selected_version)
        current_files = flatten_files(current_contents)
    except GitHubAPIError as e:
        st.error(str(e))
        st.stop()

    main_doc = choose_main_document(repo_name, current_files)
    if not main_doc:
        st.warning("No main document could be inferred for this repository and version.")
        available_files = [item.get("name", "") for item in current_files]
        if available_files:
            st.write("Available root files:")
            st.write(sorted(available_files))
        st.stop()

    render_document(token, owner, repo_name, selected_version, main_doc)


class RepoNavigatorTests(unittest.TestCase):
    def test_normalize_path(self) -> None:
        self.assertEqual(normalize_path("/src/utils/"), "src/utils")
        self.assertEqual(normalize_path(""), "")

    def test_repo_matches_name_description_and_topics(self) -> None:
        repo = {
            "full_name": "octo/demo",
            "name": "demo",
            "description": "Internal science tooling",
            "language": "Python",
            "topics": ["astronomy", "calibration"],
        }
        self.assertTrue(repo_matches(repo, "science"))
        self.assertTrue(repo_matches(repo, "python"))
        self.assertTrue(repo_matches(repo, "astronomy"))
        self.assertFalse(repo_matches(repo, "rust"))

    def test_get_token_precedence(self) -> None:
        token = get_token(
            session_state={"github_token": "session-token"},
            query_params={"token": "query-token"},
            env={"GITHUB_TOKEN": "env-token"},
        )
        self.assertEqual(token, "session-token")

    def test_get_token_falls_back_to_query_then_env(self) -> None:
        self.assertEqual(
            get_token(session_state={}, query_params={"token": "query-token"}, env={"GITHUB_TOKEN": "env-token"}),
            "query-token",
        )
        self.assertEqual(get_token(session_state={}, query_params={}, env={"GITHUB_TOKEN": "env-token"}), "env-token")

    def test_filter_repos_with_required_topic(self) -> None:
        repos = [
            {"full_name": "a/docs", "name": "docs", "private": False, "archived": False, "description": "alpha", "topics": ["roman-docs"]},
            {"full_name": "a/other", "name": "other", "private": False, "archived": False, "description": "beta", "topics": ["misc"]},
        ]
        result = filter_repos(repos, query="", repo_visibility="all", archived_mode="exclude", required_topic="roman-docs")
        self.assertEqual([r["name"] for r in result], ["docs"])

    def test_infer_main_document_name(self) -> None:
        self.assertEqual(
            infer_main_document_name("roman-ssc_d-m007"),
            ["SSC_D-M007.md", "SSC_D-M007.pdf", "SSC_D-M007.docx"],
        )
        self.assertEqual(
            infer_main_document_name("roman-ssc_d-t008"),
            ["SSC_D-T008.md", "SSC_D-T008.pdf", "SSC_D-T008.docx"],
        )
        self.assertEqual(infer_main_document_name("other-repo"), [])

    def test_choose_main_document_prefers_inferred_file(self) -> None:
        files = [
            {"name": "README.md", "type": "file"},
            {"name": "SSC_D-M007.md", "type": "file"},
            {"name": "notes.pdf", "type": "file"},
        ]
        picked = choose_main_document("roman-ssc_d-m007", files)
        self.assertIsNotNone(picked)
        self.assertEqual(picked["name"], "SSC_D-M007.md")

    def test_choose_main_document_fallback_order(self) -> None:
        files = [
            {"name": "zeta.pdf", "type": "file"},
            {"name": "alpha.docx", "type": "file"},
        ]
        picked = choose_main_document("unknown", files)
        self.assertIsNotNone(picked)
        self.assertEqual(picked["name"], "zeta.pdf")

    def test_select_default_version_prefers_latest_tag(self) -> None:
        tags = [{"name": "v1.2.0"}, {"name": "v1.10.0"}, {"name": "v1.3.0"}]
        options = build_version_options("main", branches=[{"name": "develop"}], tags=tags)
        self.assertEqual(select_default_version("main", options, tags), "v1.10.0")

    def test_select_default_version_falls_back_to_main(self) -> None:
        options = build_version_options("main", branches=[{"name": "develop"}], tags=[])
        self.assertEqual(select_default_version("main", options, []), "main")


if __name__ == "__main__":
    if "--test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        run_streamlit_app()
