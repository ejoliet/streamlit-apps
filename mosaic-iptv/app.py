# pip install streamlit

import math
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Multi-stream Viewer", layout="wide")
st.title("Multi-stream Viewer (HLS .m3u8 + regular video URLs)")

if "streams" not in st.session_state:
    # each item: {"url": str, "label": str}
    st.session_state.streams = []

with st.form("add_stream", clear_on_submit=True):
    url = st.text_input(
        "Paste a stream URL (e.g., https://fl1.moveonjoy.com/C-SPAN/index.m3u8)",
        placeholder="https://example.com/stream.m3u8",
    )
    label = st.text_input("Initial label (optional)", placeholder="C-SPAN")
    cols = st.columns([1, 1, 6])
    add = cols[0].form_submit_button("Add")
    clear = cols[1].form_submit_button("Clear all")

if clear:
    st.session_state.streams = []
    st.rerun()

if add and url.strip():
    st.session_state.streams.append({"url": url.strip(), "label": label.strip()})
    st.rerun()


def hls_player_html(m3u8_url: str, autoplay: bool, muted: bool) -> str:
    auto_attr = "autoplay" if autoplay else ""
    muted_attr = "muted" if muted else ""

    return f"""
<div style="border: 2px solid rgba(255,255,255,0.25); border-radius: 10px; padding: 8px;">
  <video id="video" controls playsinline style="width: 100%; height: auto;"
         {auto_attr} {muted_attr}></video>
</div>

<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<script>
  const video = document.getElementById('video');
  const src = {m3u8_url!r};

  video.muted = {str(muted).lower()};
  video.autoplay = {str(autoplay).lower()};

  function start() {{
    if (video.canPlayType('application/vnd.apple.mpegurl')) {{
      // Native HLS (Safari/iOS)
      video.src = src;
      if ({str(autoplay).lower()}) {{
        video.play().catch(() => {{}});
      }}
      return;
    }}

    if (window.Hls && Hls.isSupported()) {{
      const hls = new Hls({{
        enableWorker: true,
        lowLatencyMode: true
      }});
      hls.loadSource(src);
      hls.attachMedia(video);

      hls.on(Hls.Events.MANIFEST_PARSED, function() {{
        if ({str(autoplay).lower()}) {{
          video.play().catch(() => {{}});
        }}
      }});

      hls.on(Hls.Events.ERROR, function(event, data) {{
        console.log("HLS.js error:", data);
      }});
    }} else {{
      video.outerHTML = "<p>HLS is not supported in this browser.</p>";
    }}
  }}

  start();
</script>
    """


streams = st.session_state.streams
if not streams:
    st.info("Add one or more stream URLs above.")
    st.stop()

# Flexible grid columns
n = len(streams)
if n <= 2:
    ncols = 2
elif n <= 6:
    ncols = 3
elif n <= 12:
    ncols = 4
else:
    ncols = 5

rows = math.ceil(n / ncols)

for r in range(rows):
    row_cols = st.columns(ncols, gap="medium")
    for c in range(ncols):
        i = r * ncols + c
        if i >= n:
            continue

        item = streams[i]
        stream_url = item["url"]

        with row_cols[c]:
            header = st.columns([6, 1])
            # Editable label AFTER adding
            new_label = header[0].text_input(
                "Label",
                value=item.get("label") or f"Stream {i+1}",
                key=f"label_{i}",
                label_visibility="collapsed",
                placeholder="Label",
            )
            st.session_state.streams[i]["label"] = new_label

            if header[1].button("✕", key=f"del_{i}", help="Remove this stream"):
                st.session_state.streams.pop(i)
                st.rerun()

            # Per-tile controls (defaults: muted=True, autoplay=True)
            ctl = st.columns(2)
            muted = ctl[0].toggle("Muted", value=True, key=f"muted_{i}")
            autoplay = ctl[1].toggle("Autoplay", value=True, key=f"autoplay_{i}")

            # Player
            if stream_url.lower().endswith(".m3u8"):
                components.html(
                    hls_player_html(stream_url, autoplay=autoplay, muted=muted),
                    height=360,
                    scrolling=False,
                )
            else:
                st.video(stream_url)

            st.caption(stream_url)

