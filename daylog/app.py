"""
Local-first Streamlit Journal (MVP) + Auto Weather
- Per-day journal: mood, tasks (free text), notes
- Photos (stored in SQLite as bytes)
- Browse by date + simple week view
- Export entry as ICS
- Auto weather for selected day (historical or forecast) using Open-Meteo (no key)
  - Location auto-detected via IP, with manual fallback
  - Weather cached locally in SQLite per day
"""

import io
import os
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from typing import Optional, List, Tuple, Dict, Any

import requests
import streamlit as st
from PIL import Image
from ics import Calendar, Event

import calendar
from datetime import date


APP_TITLE = "DayLog — Local-first Journal"
DB_PATH = os.environ.get("JOURNAL_DB_PATH", "journal.db")

MOODS = ["", "😀 Great", "🙂 Good", "😐 Okay", "🙁 Low", "😣 Stressed", "😴 Tired"]

# -----------------------------
# SQLite
# -----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entries (
            entry_date TEXT PRIMARY KEY,          -- YYYY-MM-DD
            mood TEXT,
            tasks TEXT,                           -- newline-separated free text
            notes TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT NOT NULL,
            filename TEXT,
            mime TEXT,
            content BLOB NOT NULL,
            created_at TEXT,
            FOREIGN KEY(entry_date) REFERENCES entries(entry_date)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_cache (
            entry_date TEXT PRIMARY KEY,          -- YYYY-MM-DD
            location_name TEXT,
            latitude REAL,
            longitude REAL,
            timezone TEXT,
            tmax_c REAL,
            tmin_c REAL,
            precip_mm REAL,
            windmax_kmh REAL,
            weathercode INTEGER,
            sunrise TEXT,
            sunset TEXT,
            fetched_at TEXT
        );
        """
    )
    conn.commit()

@dataclass
class Entry:
    entry_date: str
    mood: str
    tasks: str
    notes: str
    created_at: str
    updated_at: str

def upsert_entry(conn: sqlite3.Connection, entry_date: str, mood: str, tasks: str, notes: str):
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn.execute(
        """
        INSERT INTO entries(entry_date, mood, tasks, notes, created_at, updated_at)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(entry_date) DO UPDATE SET
            mood=excluded.mood,
            tasks=excluded.tasks,
            notes=excluded.notes,
            updated_at=excluded.updated_at;
        """,
        (entry_date, mood, tasks, notes, now, now),
    )
    conn.commit()

def get_entry(conn: sqlite3.Connection, entry_date: str) -> Optional[Entry]:
    cur = conn.execute(
        "SELECT entry_date, mood, tasks, notes, created_at, updated_at FROM entries WHERE entry_date=?",
        (entry_date,),
    )
    row = cur.fetchone()
    return Entry(*row) if row else None

def list_entries_in_range(conn: sqlite3.Connection, start_date: str, end_date: str) -> List[Entry]:
    cur = conn.execute(
        """
        SELECT entry_date, mood, tasks, notes, created_at, updated_at
        FROM entries
        WHERE entry_date BETWEEN ? AND ?
        ORDER BY entry_date DESC
        """,
        (start_date, end_date),
    )
    return [Entry(*r) for r in cur.fetchall()]

def add_photo(conn: sqlite3.Connection, entry_date: str, filename: str, mime: str, content: bytes):
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn.execute(
        """
        INSERT INTO photos(entry_date, filename, mime, content, created_at)
        VALUES(?, ?, ?, ?, ?)
        """,
        (entry_date, filename, mime, content, now),
    )
    conn.commit()

def list_photos(conn: sqlite3.Connection, entry_date: str) -> List[Tuple[int, str, str, bytes, str]]:
    cur = conn.execute(
        "SELECT id, filename, mime, content, created_at FROM photos WHERE entry_date=? ORDER BY id DESC",
        (entry_date,),
    )
    return cur.fetchall()

def delete_photo(conn: sqlite3.Connection, photo_id: int):
    conn.execute("DELETE FROM photos WHERE id=?", (photo_id,))
    conn.commit()

def get_weather_cache(conn: sqlite3.Connection, entry_date: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT entry_date, location_name, latitude, longitude, timezone,
               tmax_c, tmin_c, precip_mm, windmax_kmh, weathercode, sunrise, sunset, fetched_at
        FROM weather_cache
        WHERE entry_date=?
        """,
        (entry_date,),
    )
    row = cur.fetchone()
    if not row:
        return None
    keys = [
        "entry_date","location_name","latitude","longitude","timezone",
        "tmax_c","tmin_c","precip_mm","windmax_kmh","weathercode","sunrise","sunset","fetched_at"
    ]
    return dict(zip(keys, row))

def upsert_weather_cache(conn: sqlite3.Connection, entry_date: str, payload: Dict[str, Any]):
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn.execute(
        """
        INSERT INTO weather_cache(
            entry_date, location_name, latitude, longitude, timezone,
            tmax_c, tmin_c, precip_mm, windmax_kmh, weathercode, sunrise, sunset, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(entry_date) DO UPDATE SET
            location_name=excluded.location_name,
            latitude=excluded.latitude,
            longitude=excluded.longitude,
            timezone=excluded.timezone,
            tmax_c=excluded.tmax_c,
            tmin_c=excluded.tmin_c,
            precip_mm=excluded.precip_mm,
            windmax_kmh=excluded.windmax_kmh,
            weathercode=excluded.weathercode,
            sunrise=excluded.sunrise,
            sunset=excluded.sunset,
            fetched_at=excluded.fetched_at;
        """,
        (
            entry_date,
            payload.get("location_name"),
            payload.get("latitude"),
            payload.get("longitude"),
            payload.get("timezone"),
            payload.get("tmax_c"),
            payload.get("tmin_c"),
            payload.get("precip_mm"),
            payload.get("windmax_kmh"),
            payload.get("weathercode"),
            payload.get("sunrise"),
            payload.get("sunset"),
            now,
        ),
    )
    conn.commit()

# -----------------------------
# Weather (Open-Meteo)
# -----------------------------
def ip_geolocate() -> Optional[Dict[str, Any]]:
    """
    Best-effort, zero-config location via IP.
    If this fails (VPN, blocked outbound), we fall back to manual city.
    """
    try:
        r = requests.get("https://ipapi.co/json/", timeout=3)
        r.raise_for_status()
        j = r.json()
        lat = j.get("latitude")
        lon = j.get("longitude")
        city = j.get("city")
        region = j.get("region")
        country = j.get("country_name")
        if lat is None or lon is None:
            return None
        name = ", ".join([p for p in [city, region, country] if p])
        return {"latitude": float(lat), "longitude": float(lon), "location_name": name or "Current location"}
    except Exception:
        return None

def geocode_city(city: str) -> Optional[Dict[str, Any]]:
    """
    Open-Meteo geocoding (no key).
    """
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=5,
        )
        r.raise_for_status()
        j = r.json()
        results = j.get("results") or []
        if not results:
            return None
        top = results[0]
        name = ", ".join([p for p in [top.get("name"), top.get("admin1"), top.get("country")] if p])
        return {"latitude": float(top["latitude"]), "longitude": float(top["longitude"]), "location_name": name}
    except Exception:
        return None

def fetch_daily_weather(lat: float, lon: float, day: date) -> Optional[Dict[str, Any]]:
    """
    Pull daily aggregates for a single date.
    Uses Open-Meteo forecast endpoint which supports past days (within their supported range)
    and future forecasts.
    """
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "timezone": "auto",
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max",
                "weathercode",
                "sunrise",
                "sunset",
            ]),
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=8)
        r.raise_for_status()
        j = r.json()
        daily = j.get("daily") or {}
        # All daily arrays should have length 1 for single-day request
        def one(key, default=None):
            arr = daily.get(key)
            return arr[0] if isinstance(arr, list) and arr else default

        return {
            "latitude": j.get("latitude"),
            "longitude": j.get("longitude"),
            "timezone": j.get("timezone"),
            "tmax_c": one("temperature_2m_max"),
            "tmin_c": one("temperature_2m_min"),
            "precip_mm": one("precipitation_sum"),
            "windmax_kmh": one("wind_speed_10m_max"),
            "weathercode": one("weathercode"),
            "sunrise": one("sunrise"),
            "sunset": one("sunset"),
        }
    except Exception:
        return None

WEATHER_CODE_HINTS = {
    0: "Clear",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Violent showers",
    95: "Thunderstorm",
    96: "Thunderstorm + hail",
    99: "Thunderstorm + hail",
}

# -----------------------------
# ICS Export
# -----------------------------
def build_ics_for_entry(entry: Entry, local_day: date) -> bytes:
    cal = Calendar()
    e = Event()
    e.name = f"Journal — {local_day.isoformat()} ({entry.mood or 'no mood'})"
    # simple placeholder block
    e.begin = datetime.combine(local_day, dtime(19, 0))
    e.end = datetime.combine(local_day, dtime(19, 30))
    parts = []
    if entry.mood:
        parts.append(f"Mood: {entry.mood}")
    if entry.tasks.strip():
        parts.append("Tasks:\n" + entry.tasks.strip())
    if entry.notes.strip():
        parts.append("Notes:\n" + entry.notes.strip())
    e.description = "\n\n".join(parts).strip()
    cal.events.add(e)
    return str(cal).encode("utf-8")

# -----------------------------
# Helpers
# -----------------------------
def week_bounds(d: date) -> Tuple[date, date]:
    start = d - timedelta(days=d.weekday())  # Mon
    end = start + timedelta(days=6)          # Sun
    return start, end

def zip_db(db_path: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(db_path, arcname=os.path.basename(db_path))
    return buf.getvalue()

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Local-first journaling: everything stored in a local SQLite file (journal.db).")

conn = get_conn()
init_db(conn)

# Session state
if "selected_day" not in st.session_state:
    st.session_state["selected_day"] = date.today()
if "loc" not in st.session_state:
    st.session_state["loc"] = ip_geolocate()  # best-effort

# Sidebar: backups + location
st.sidebar.header("Local-first")
if os.path.exists(DB_PATH):
    st.sidebar.download_button(
        "Download backup (journal.db.zip)",
        data=zip_db(DB_PATH),
        file_name="journal.db.zip",
        mime="application/zip",
        use_container_width=True,
    )

st.sidebar.header("Weather location")
loc = st.session_state["loc"]
if loc:
    st.sidebar.write(f"Auto: **{loc.get('location_name','(unknown)')}**")
else:
    st.sidebar.warning("Auto-location failed (common with VPN/firewall).")

manual_city = st.sidebar.text_input("Fallback city (optional)", value="")
if st.sidebar.button("Use fallback city"):
    g = geocode_city(manual_city.strip()) if manual_city.strip() else None
    if g:
        st.session_state["loc"] = g
        st.sidebar.success(f"Using: {g['location_name']}")
        st.rerun()
    else:
        st.sidebar.error("Could not geocode that city.")

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Pick a day")
    selected_day: date = st.date_input("Journal date", value=st.session_state["selected_day"])
    st.session_state["selected_day"] = selected_day
    selected_key = selected_day.isoformat()

    st.divider()
    st.subheader("Recent days")
    recent = list_entries_in_range(conn, (date.today() - timedelta(days=30)).isoformat(), date.today().isoformat())
    if recent:
        for e in recent[:30]:
            if st.button(f"{e.entry_date} · {e.mood or 'no mood'}", key=f"jump_{e.entry_date}"):
                st.session_state["selected_day"] = datetime.fromisoformat(e.entry_date).date()
                st.rerun()
    else:
        st.caption("No entries yet.")

    st.divider()
    st.subheader("Week view")
    ws, we = week_bounds(selected_day)
    st.write(f"**{ws.isoformat()} → {we.isoformat()}**")
    week_entries = list_entries_in_range(conn, ws.isoformat(), we.isoformat())
    if not week_entries:
        st.caption("No entries this week yet.")
    else:
        for e in week_entries:
            preview = (e.notes or "").strip().replace("\n", " ")
            if len(preview) > 90:
                preview = preview[:90] + "…"
            st.markdown(f"- **{e.entry_date}** — {e.mood or 'no mood'}" + (f" — {preview}" if preview else ""))

with col_right:
    st.subheader(f"Entry for {selected_key}")

    existing = get_entry(conn, selected_key)
    default_mood = existing.mood if existing else ""
    default_tasks = existing.tasks if existing else ""
    default_notes = existing.notes if existing else ""

    # Weather panel (auto + cached)
    with st.container(border=True):
        st.markdown("### Weather (auto)")
        cached = get_weather_cache(conn, selected_key)

        # If no cache (or user wants refresh), try fetching
        refresh = st.button("Refresh weather", use_container_width=False)
        if refresh:
            cached = None

        if cached is None:
            if st.session_state["loc"]:
                lat = st.session_state["loc"]["latitude"]
                lon = st.session_state["loc"]["longitude"]
                w = fetch_daily_weather(lat, lon, selected_day)
                if w:
                    payload = {
                        **w,
                        "location_name": st.session_state["loc"].get("location_name"),
                    }
                    upsert_weather_cache(conn, selected_key, payload)
                    cached = get_weather_cache(conn, selected_key)
                else:
                    st.warning("Couldn’t fetch weather (network blocked or provider error).")
            else:
                st.info("Set a fallback city in the sidebar to enable weather.")

        if cached:
            code = cached.get("weathercode")
            hint = WEATHER_CODE_HINTS.get(code, "—")
            st.write(f"**Location:** {cached.get('location_name') or '—'}")
            st.write(
                f"**Conditions:** {hint} (code {code})  ·  "
                f"**High/Low:** {cached.get('tmax_c')}°C / {cached.get('tmin_c')}°C  ·  "
                f"**Precip:** {cached.get('precip_mm')} mm  ·  "
                f"**Wind max:** {cached.get('windmax_kmh')} km/h"
            )
            st.caption(f"Sunrise: {cached.get('sunrise')} · Sunset: {cached.get('sunset')}")

    # Editor
    mood = st.selectbox(
        "Mood",
        options=MOODS,
        index=MOODS.index(default_mood) if default_mood in MOODS else 0,
    )
    tasks = st.text_area("Tasks accomplished (free text, one per line if you want)", value=default_tasks, height=120)
    notes = st.text_area("Notes", value=default_notes, height=220)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Save", use_container_width=True):
            upsert_entry(conn, selected_key, mood, tasks, notes)
            st.success("Saved.")
            st.rerun()

    with c2:
        entry_now = get_entry(conn, selected_key)
        if entry_now and (entry_now.mood or entry_now.tasks.strip() or entry_now.notes.strip()):
            ics_bytes = build_ics_for_entry(entry_now, selected_day)
            st.download_button(
                "Export ICS",
                data=ics_bytes,
                file_name=f"journal-{selected_key}.ics",
                mime="text/calendar",
                use_container_width=True,
            )
        else:
            st.button("Export ICS", disabled=True, use_container_width=True)

    with c3:
        st.caption("Import the ICS into your calendar if you want a ‘journal marker’ for that day.")

    st.divider()
    st.subheader("Photos")

    uploaded = st.file_uploader(
        "Add photos (jpg/png/webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if uploaded:
        # Ensure entry exists so photos reference a day (even if text is empty)
        if not get_entry(conn, selected_key):
            upsert_entry(conn, selected_key, mood, tasks, notes)

        for f in uploaded:
            content = f.read()
            add_photo(conn, selected_key, f.name, f.type or "application/octet-stream", content)
        st.success(f"Uploaded {len(uploaded)} photo(s).")
        st.rerun()

    photos = list_photos(conn, selected_key)
    if not photos:
        st.caption("No photos for this day yet.")
    else:
        grid_cols = st.columns(3)
        for idx, (pid, filename, mime, blob, created_at) in enumerate(photos):
            with grid_cols[idx % 3]:
                try:
                    img = Image.open(io.BytesIO(blob))
                    st.image(img, caption=filename, use_container_width=True)
                except Exception:
                    st.warning(f"Could not render {filename} ({mime}).")
                if st.button(f"Delete ({pid})", key=f"del_{pid}"):
                    delete_photo(conn, pid)
                    st.rerun()

    st.divider()
    st.subheader("Search")
    q = st.text_input("Find text in notes/tasks (local)", value="")
    if q.strip():
        cur = conn.execute(
            """
            SELECT entry_date, mood, tasks, notes
            FROM entries
            WHERE tasks LIKE ? OR notes LIKE ?
            ORDER BY entry_date DESC
            LIMIT 50
            """,
            (f"%{q}%", f"%{q}%"),
        )
        rows = cur.fetchall()
        if not rows:
            st.caption("No matches.")
        else:
            for d, m, t, n in rows:
                st.markdown(f"- **{d}** — {m or 'no mood'}")
