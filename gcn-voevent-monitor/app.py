"""GCN VOEvent Streamlit dashboard.

Run with:
    streamlit run app.py
"""

import time
import pandas as pd
import plotly.express as px
import streamlit as st

import db
import gcn_consumer

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GCN VOEvent Monitor",
    page_icon="telescope",
    layout="wide",
)

# ── ensure DB exists and consumer is running ───────────────────────────────────
db.init_db()
gcn_consumer.start()

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("GCN VOEvent Monitor")

    status = gcn_consumer.get_status()
    if status["running"]:
        st.success("Consumer: running")
    else:
        st.error("Consumer: stopped")
        if st.button("Restart consumer"):
            gcn_consumer.start()

    st.metric("Events received (session)", status["received"])
    if status["last_event_time"]:
        st.caption(f"Last event: {status['last_event_time']}")
    if status["last_error"]:
        st.warning(f"Last error: {status['last_error']}")

    st.divider()

    window_hours = st.select_slider(
        "Display window",
        options=[1, 3, 6, 12, 24],
        value=24,
        format_func=lambda h: f"Last {h}h",
    )

    topic_filter = st.text_input(
        "Filter by topic (substring)",
        placeholder="e.g. FERMI, SWIFT, LVC",
    )

    auto_refresh = st.checkbox("Auto-refresh (30 s)", value=True)
    if auto_refresh:
        time.sleep(0.1)  # yield so the checkbox state is read
        st.caption("Page refreshes every 30 s")

# ── fetch data ─────────────────────────────────────────────────────────────────
events = db.query_events(hours=window_hours, limit=2000)
df = pd.DataFrame(events) if events else pd.DataFrame(
    columns=["id","ivorn","topic","role","received_at","event_time",
             "author","description","ra","dec","error_radius","raw_xml"]
)

if not df.empty and topic_filter:
    mask = df["topic"].str.contains(topic_filter, case=False, na=False)
    df = df[mask]

# ── top metrics ────────────────────────────────────────────────────────────────
total_db   = db.count_events()
window_cnt = len(df)
with_pos   = int(df["ra"].notna().sum()) if not df.empty else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"Events (last {window_hours}h)", window_cnt)
col2.metric("With sky position", with_pos)
col3.metric("Total in DB", total_db)
col4.metric("Topics active", int(df["topic"].nunique()) if not df.empty else 0)

st.divider()

# ── time-series bar chart ──────────────────────────────────────────────────────
st.subheader(f"Events per hour — last {window_hours}h")

if not df.empty:
    df["hour"] = pd.to_datetime(df["received_at"]).dt.floor("h")
    hourly = df.groupby("hour").size().reset_index(name="count")
    fig_ts = px.bar(hourly, x="hour", y="count", labels={"hour": "UTC hour", "count": "Events"})
    fig_ts.update_layout(margin=dict(t=20, b=20), height=220)
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("No events yet in this window.")

# ── topic breakdown + sky map ──────────────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.subheader("Events by topic")
    if not df.empty:
        topic_counts = (
            df.groupby("topic").size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(20)
        )
        # short label: strip common prefix
        topic_counts["label"] = topic_counts["topic"].str.replace(
            "gcn.classic.voevent.", "", regex=False
        ).str.replace("gcn.notices.svom.voevent.", "SVOM/", regex=False)
        fig_bar = px.bar(
            topic_counts,
            x="count",
            y="label",
            orientation="h",
            labels={"label": "", "count": "Count"},
        )
        fig_bar.update_layout(margin=dict(t=10, b=10), height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No data.")

with right:
    st.subheader("Sky positions (events with RA/Dec)")
    sky_df = df.dropna(subset=["ra", "dec"]) if not df.empty else pd.DataFrame()
    if not sky_df.empty:
        sky_df = sky_df.copy()
        sky_df["label"] = sky_df["topic"].str.replace("gcn.classic.voevent.", "", regex=False)
        fig_sky = px.scatter(
            sky_df,
            x="ra",
            y="dec",
            color="label",
            hover_data={"ivorn": True, "received_at": True, "error_radius": True, "label": False},
            labels={"ra": "RA (deg)", "dec": "Dec (deg)", "label": "Topic"},
            title="",
        )
        fig_sky.update_xaxes(range=[360, 0])  # RA increases right-to-left
        fig_sky.update_layout(margin=dict(t=10, b=10), height=400)
        st.plotly_chart(fig_sky, use_container_width=True)
    else:
        st.info("No events with sky coordinates in this window.")

# ── event table ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Recent events")

if not df.empty:
    display_cols = ["received_at", "topic", "role", "ivorn", "ra", "dec", "error_radius", "description"]
    table_df = df[display_cols].copy()
    table_df["topic"] = table_df["topic"].str.replace("gcn.classic.voevent.", "", regex=False)
    table_df = table_df.rename(columns={
        "received_at": "Received (UTC)",
        "topic": "Topic",
        "role": "Role",
        "ivorn": "IVORN",
        "ra": "RA",
        "dec": "Dec",
        "error_radius": "Err (deg)",
        "description": "Description",
    })
    st.dataframe(table_df, use_container_width=True, height=400)

    # detail expander for raw XML
    st.subheader("Raw VOEvent XML")
    ivorn_list = df["ivorn"].tolist()
    selected = st.selectbox("Select event by IVORN", ivorn_list)
    if selected:
        row = df[df["ivorn"] == selected].iloc[0]
        st.code(row.get("raw_xml", ""), language="xml")
else:
    st.info("Waiting for events...")

# ── auto-refresh ───────────────────────────────────────────────────────────────
if auto_refresh:
    st.empty()
    time.sleep(30)
    st.rerun()
