# app.py
# pip install streamlit requests beautifulsoup4 lxml pandas

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
import streamlit.components.v1 as components  # recommended import style for iframe  [oai_citation:3‡Streamlit Docs](https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v1.iframe?utm_source=chatgpt.com)


BASE = "https://www.justice.gov"
DATASET_PAGES = {
    "Data Set 1": f"{BASE}/epstein/doj-disclosures/data-set-1-files",
    "Data Set 2": f"{BASE}/epstein/doj-disclosures/data-set-2-files",
    "Data Set 3": f"{BASE}/epstein/doj-disclosures/data-set-3-files",
    "Data Set 4": f"{BASE}/epstein/doj-disclosures/data-set-4-files",
    "Data Set 5": f"{BASE}/epstein/doj-disclosures/data-set-5-files",
    "Data Set 6": f"{BASE}/epstein/doj-disclosures/data-set-6-files",
    "Data Set 7": f"{BASE}/epstein/doj-disclosures/data-set-7-files",
}

PDF_RE = re.compile(r"\.pdf(\?|$)", re.IGNORECASE)


@dataclass(frozen=True)
class PdfItem:
    dataset: str
    title: str
    url: str
    source_page: str


def _get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def _normalize_url(href: str, page_url: str) -> str:
    return urljoin(page_url, href)


def _is_justice_pdf(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.netloc == urlparse(BASE).netloc and bool(PDF_RE.search(p.path))
    except Exception:
        return False


def _find_next_page(soup: BeautifulSoup, current_url: str) -> str | None:
    # DOJ pages typically have a pager link labeled "Next"
    nxt = soup.find("a", string=re.compile(r"^\s*Next\s*$", re.IGNORECASE))
    if nxt and nxt.get("href"):
        return _normalize_url(nxt["href"], current_url)
    return None


def scrape_dataset(session: requests.Session, label: str, start_url: str) -> list[PdfItem]:
    out: list[PdfItem] = []
    seen: set[str] = set()

    url: str | None = start_url
    while url:
        soup = _get_soup(session, url)

        for a in soup.select("a[href]"):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            if not PDF_RE.search(href):
                continue

            pdf_url = _normalize_url(href, url)
            if not _is_justice_pdf(pdf_url):
                continue
            if pdf_url in seen:
                continue
            seen.add(pdf_url)

            title = a.get_text(strip=True) or pdf_url.split("/")[-1]
            out.append(PdfItem(dataset=label, title=title, url=pdf_url, source_page=url))

        url = _find_next_page(soup, url)
        time.sleep(0.2)  # be polite

    return out


@st.cache_data(ttl=60 * 30)  # cache for 30 minutes
def load_all_pdfs(selected: list[str]) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (compatible; streamlit-epstein-pdf-browser/1.0)"}
    )

    items: list[PdfItem] = []
    for ds in selected:
        items.extend(scrape_dataset(session, ds, DATASET_PAGES[ds]))

    df = pd.DataFrame([i.__dict__ for i in items]).drop_duplicates(subset=["url"])
    df = df.sort_values(["dataset", "title"]).reset_index(drop=True)
    return df


def main() -> None:
    st.set_page_config(page_title="DOJ Epstein PDFs – Carousel + Grid", layout="wide")
    st.title("DOJ Epstein Disclosures – PDF Browser")

    with st.sidebar:
        st.header("Controls")
        datasets = st.multiselect(
            "Data sets",
            options=list(DATASET_PAGES.keys()),
            default=list(DATASET_PAGES.keys()),
            help="These are the 'Data Set N Files' pages under DOJ Disclosures.",
        )

        q = st.text_input("Search", placeholder="EFTA0000… or any text")
        cols = st.slider("Grid columns", min_value=2, max_value=10, value=5, step=1)
        iframe_h = st.slider("PDF viewer height", min_value=500, max_value=1400, value=900, step=50)

        st.divider()
        st.caption("Tip: If a PDF won’t embed due to browser restrictions, use the “Open PDF” link.")

        reload_now = st.button("Refresh scrape (ignore cache)")

    if not datasets:
        st.info("Select at least one data set in the sidebar.")
        return

    if reload_now:
        load_all_pdfs.clear()

    df = load_all_pdfs(datasets)
    if q.strip():
        qq = q.strip().lower()
        df = df[df["title"].str.lower().str.contains(qq) | df["url"].str.lower().str.contains(qq)]

    st.caption(f"Found **{len(df)}** PDFs across {len(datasets)} selected data sets.")

    # --- Carousel state ---
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    if len(df) == 0:
        st.warning("No matches. Try clearing the search.")
        return

    st.subheader("Carousel (flip one-by-one)")
    c1, c2, c3, c4 = st.columns([1, 1, 6, 2])  # layout helper  [oai_citation:4‡Streamlit Docs](https://docs.streamlit.io/develop/api-reference/layout/st.columns?utm_source=chatgpt.com)
    with c1:
        if st.button("◀ Prev"):
            st.session_state.idx = (st.session_state.idx - 1) % len(df)
    with c2:
        if st.button("Next ▶"):
            st.session_state.idx = (st.session_state.idx + 1) % len(df)

    idx = int(st.session_state.idx) % len(df)
    row = df.iloc[idx].to_dict()

    with c3:
        st.markdown(f"**{row['title']}**  \n*{row['dataset']}*")
        st.markdown(f"[Open PDF]({row['url']})")

    with c4:
        st.metric("Item", f"{idx+1} / {len(df)}")

    # Embed PDF (iframe)
    # Streamlit's iframe helper loads a remote URL in an iframe  [oai_citation:5‡Streamlit Docs](https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v1.iframe?utm_source=chatgpt.com)
    components.iframe(row["url"], height=iframe_h, scrolling=True)

    # --- Grid ---
    st.subheader("Grid")
    st.caption("Click any card’s button to load it into the carousel viewer above.")

    cards = df.to_dict(orient="records")
    for start in range(0, len(cards), cols):
        row_items = cards[start : start + cols]
        col_containers = st.columns(cols)  #  [oai_citation:6‡Streamlit Docs](https://docs.streamlit.io/develop/api-reference/layout/st.columns?utm_source=chatgpt.com)
        for i, it in enumerate(row_items):
            with col_containers[i]:
                st.markdown(f"**{it['title']}**")
                st.caption(it["dataset"])
                st.markdown(f"[Open PDF]({it['url']})")
                if st.button("View in carousel", key=f"view_{it['url']}"):
                    # jump carousel to this URL
                    match_idx = df.index[df["url"] == it["url"]]
                    if len(match_idx):
                        st.session_state.idx = int(match_idx[0])
                        st.rerun()

    with st.expander("Show table (copy/export)"):
        st.dataframe(df, use_container_width=True)

    st.caption(
        "Source pages: DOJ Disclosures Data Set pages on justice.gov (example: Data Set 1 Files)."
    )  #  [oai_citation:7‡Department of Justice](https://www.justice.gov/epstein/doj-disclosures/data-set-1-files?utm_source=chatgpt.com)


if __name__ == "__main__":
    main()
