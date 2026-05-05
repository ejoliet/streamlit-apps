"""Background Kafka consumer thread for GCN VOEvents.

Credentials are read from environment variables GCN_CLIENT_ID and
GCN_CLIENT_SECRET (set them in a .env file or your shell — never commit them).
"""

import os
import threading
import logging
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env in the app directory if present
except ImportError:
    pass  # python-dotenv is optional; set env vars manually if not installed

import db

log = logging.getLogger(__name__)

# ── credentials from environment ──────────────────────────────────────────────
CLIENT_ID     = os.environ.get("GCN_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("GCN_CLIENT_SECRET", "")

# ── full topic list from consumer.py ──────────────────────────────────────────
TOPICS = [
    "gcn.classic.voevent.AGILE_GRB_GROUND",
    "gcn.classic.voevent.AGILE_GRB_POS_TEST",
    "gcn.classic.voevent.AGILE_GRB_REFINED",
    "gcn.classic.voevent.AGILE_GRB_WAKEUP",
    "gcn.classic.voevent.AGILE_MCAL_ALERT",
    "gcn.classic.voevent.AGILE_POINTDIR",
    "gcn.classic.voevent.AGILE_TRANS",
    "gcn.classic.voevent.AMON_ICECUBE_COINC",
    "gcn.classic.voevent.AMON_ICECUBE_EHE",
    "gcn.classic.voevent.AMON_ICECUBE_HESE",
    "gcn.classic.voevent.AMON_NU_EM_COINC",
    "gcn.classic.voevent.CALET_GBM_FLT_LC",
    "gcn.classic.voevent.CALET_GBM_GND_LC",
    "gcn.classic.voevent.FERMI_GBM_ALERT",
    "gcn.classic.voevent.FERMI_GBM_FIN_POS",
    "gcn.classic.voevent.FERMI_GBM_FLT_POS",
    "gcn.classic.voevent.FERMI_GBM_GND_POS",
    "gcn.classic.voevent.FERMI_GBM_LC",
    "gcn.classic.voevent.FERMI_GBM_POS_TEST",
    "gcn.classic.voevent.FERMI_GBM_SUBTHRESH",
    "gcn.classic.voevent.FERMI_GBM_TRANS",
    "gcn.classic.voevent.FERMI_LAT_GND",
    "gcn.classic.voevent.FERMI_LAT_MONITOR",
    "gcn.classic.voevent.FERMI_LAT_OFFLINE",
    "gcn.classic.voevent.FERMI_LAT_POS_DIAG",
    "gcn.classic.voevent.FERMI_LAT_POS_INI",
    "gcn.classic.voevent.FERMI_LAT_POS_TEST",
    "gcn.classic.voevent.FERMI_LAT_POS_UPD",
    "gcn.classic.voevent.FERMI_LAT_TRANS",
    "gcn.classic.voevent.FERMI_POINTDIR",
    "gcn.classic.voevent.FERMI_SC_SLEW",
    "gcn.classic.voevent.GECAM_FLT",
    "gcn.classic.voevent.GECAM_GND",
    "gcn.classic.voevent.ICECUBE_ASTROTRACK_BRONZE",
    "gcn.classic.voevent.ICECUBE_ASTROTRACK_GOLD",
    "gcn.classic.voevent.ICECUBE_CASCADE",
    "gcn.classic.voevent.INTEGRAL_OFFLINE",
    "gcn.classic.voevent.INTEGRAL_POINTDIR",
    "gcn.classic.voevent.INTEGRAL_REFINED",
    "gcn.classic.voevent.INTEGRAL_SPIACS",
    "gcn.classic.voevent.INTEGRAL_WAKEUP",
    "gcn.classic.voevent.INTEGRAL_WEAK",
    "gcn.classic.voevent.IPN_POS",
    "gcn.classic.voevent.IPN_RAW",
    "gcn.classic.voevent.IPN_SEG",
    "gcn.classic.voevent.LVC_COUNTERPART",
    "gcn.classic.voevent.LVC_EARLY_WARNING",
    "gcn.classic.voevent.LVC_INITIAL",
    "gcn.classic.voevent.LVC_PRELIMINARY",
    "gcn.classic.voevent.LVC_RETRACTION",
    "gcn.classic.voevent.LVC_TEST",
    "gcn.classic.voevent.LVC_UPDATE",
    "gcn.classic.voevent.MAXI_KNOWN",
    "gcn.classic.voevent.MAXI_TEST",
    "gcn.classic.voevent.MAXI_UNKNOWN",
    "gcn.classic.voevent.SWIFT_ACTUAL_POINTDIR",
    "gcn.classic.voevent.SWIFT_BAT_GRB_LC",
    "gcn.classic.voevent.SWIFT_BAT_GRB_LC_PROC",
    "gcn.classic.voevent.SWIFT_BAT_GRB_POS_ACK",
    "gcn.classic.voevent.SWIFT_BAT_GRB_POS_NACK",
    "gcn.classic.voevent.SWIFT_BAT_GRB_POS_TEST",
    "gcn.classic.voevent.SWIFT_BAT_KNOWN_SRC",
    "gcn.classic.voevent.SWIFT_BAT_MONITOR",
    "gcn.classic.voevent.SWIFT_BAT_QL_POS",
    "gcn.classic.voevent.SWIFT_BAT_SCALEDMAP",
    "gcn.classic.voevent.SWIFT_FOM_OBS",
    "gcn.classic.voevent.SWIFT_FOM_SAFE_POINT",
    "gcn.classic.voevent.SWIFT_FOM_SLEW_ABORT",
    "gcn.classic.voevent.SWIFT_POINTDIR",
    "gcn.classic.voevent.SWIFT_SC_SLEW",
    "gcn.classic.voevent.SWIFT_TOO_FOM",
    "gcn.classic.voevent.SWIFT_TOO_SC_SLEW",
    "gcn.classic.voevent.SWIFT_UVOT_DBURST",
    "gcn.classic.voevent.SWIFT_UVOT_DBURST_PROC",
    "gcn.classic.voevent.SWIFT_UVOT_FCHART",
    "gcn.classic.voevent.SWIFT_UVOT_FCHART_PROC",
    "gcn.classic.voevent.SWIFT_UVOT_POS",
    "gcn.classic.voevent.SWIFT_UVOT_POS_NACK",
    "gcn.classic.voevent.SWIFT_XRT_CENTROID",
    "gcn.classic.voevent.SWIFT_XRT_IMAGE",
    "gcn.classic.voevent.SWIFT_XRT_IMAGE_PROC",
    "gcn.classic.voevent.SWIFT_XRT_LC",
    "gcn.classic.voevent.SWIFT_XRT_POSITION",
    "gcn.classic.voevent.SWIFT_XRT_SPECTRUM",
    "gcn.classic.voevent.SWIFT_XRT_SPECTRUM_PROC",
    "gcn.classic.voevent.SWIFT_XRT_SPER",
    "gcn.classic.voevent.SWIFT_XRT_SPER_PROC",
    "gcn.classic.voevent.SWIFT_XRT_THRESHPIX",
    "gcn.classic.voevent.SWIFT_XRT_THRESHPIX_PROC",
    "gcn.classic.voevent.AAVSO",
    "gcn.classic.voevent.ALEXIS_SRC",
    "gcn.classic.voevent.BRAD_COORDS",
    "gcn.classic.voevent.CBAT",
    "gcn.classic.voevent.COINCIDENCE",
    "gcn.classic.voevent.COMPTEL_SRC",
    "gcn.classic.voevent.DOW_TOD",
    "gcn.classic.voevent.GRB_CNTRPART",
    "gcn.classic.voevent.GRB_COORDS",
    "gcn.classic.voevent.GRB_FINAL",
    "gcn.classic.voevent.GWHEN_COINC",
    "gcn.classic.voevent.HAWC_BURST_MONITOR",
    "gcn.classic.voevent.HUNTS_SRC",
    "gcn.classic.voevent.KONUS_LC",
    "gcn.classic.voevent.MAXBC",
    "gcn.classic.voevent.MILAGRO_POS",
    "gcn.classic.voevent.MOA",
    "gcn.classic.voevent.OGLE",
    "gcn.classic.voevent.SIMBADNED",
    "gcn.classic.voevent.SK_SN",
    "gcn.classic.voevent.SNEWS",
    "gcn.classic.voevent.SUZAKU_LC",
    "gcn.classic.voevent.TEST_COORDS",
    "gcn.notices.svom.voevent.grm",
    "gcn.notices.svom.voevent.eclairs",
    "gcn.notices.svom.voevent.mxt",
]

# ── state ──────────────────────────────────────────────────────────────────────
_thread: threading.Thread | None = None
_stop_event = threading.Event()
_status: dict = {
    "running": False,
    "received": 0,
    "last_error": None,
    "last_event_time": None,
}
_lock = threading.Lock()


# ── VOEvent parser ─────────────────────────────────────────────────────────────

def _find(root: ET.Element, *tags: str) -> ET.Element | None:
    """Try multiple tag variants (with/without namespace prefixes)."""
    STC = "http://www.ivoa.net/xml/STC/stc-v1.30.xsd"
    for tag in tags:
        el = root.find(f".//{{{STC}}}{tag}") or root.find(f".//{tag}")
        if el is not None:
            return el
    return None


def _parse_voevent(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)
    ivorn = root.get("ivorn", "")
    role  = root.get("role", "")

    # Who block
    who = root.find("Who")
    author = ""
    event_time = None
    if who is not None:
        ae = who.find("Author")
        if ae is not None:
            parts = [ae.findtext("contactName") or "", ae.findtext("shortName") or ""]
            author = " / ".join(p for p in parts if p)
        de = who.find("Date")
        if de is not None:
            event_time = (de.text or "").strip() or None

    # What → Description
    what = root.find("What")
    description = ""
    if what is not None:
        desc = what.find("Description")
        if desc is not None:
            description = (desc.text or "").strip()

    # Position
    ra = dec = error_radius = None
    try:
        c1 = _find(root, "C1")
        c2 = _find(root, "C2")
        er = _find(root, "Error2Radius")
        if c1 is not None and c1.text:
            ra = float(c1.text)
        if c2 is not None and c2.text:
            dec = float(c2.text)
        if er is not None and er.text:
            error_radius = float(er.text)
    except (TypeError, ValueError):
        pass

    return {
        "ivorn": ivorn,
        "role": role,
        "author": author,
        "event_time": event_time,
        "description": description,
        "ra": ra,
        "dec": dec,
        "error_radius": error_radius,
    }


# ── consumer loop ──────────────────────────────────────────────────────────────

def _loop() -> None:
    from gcn_kafka import Consumer

    if not CLIENT_ID or not CLIENT_SECRET:
        msg = (
            "GCN_CLIENT_ID and GCN_CLIENT_SECRET must be set. "
            "Copy .env.example to .env and fill in your credentials."
        )
        log.error(msg)
        with _lock:
            _status["last_error"] = msg
        return

    db.init_db()

    consumer = Consumer(
        {"group.id": "gcn-streamlit-monitor", "auto.offset.reset": "latest"},
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )

    try:
        consumer.subscribe(TOPICS)
        log.info("Subscribed to %d GCN topics", len(TOPICS))
        with _lock:
            _status["running"] = True

        insert_count = 0
        while not _stop_event.is_set():
            for msg in consumer.consume(timeout=1):
                if _stop_event.is_set():
                    break
                if msg.error():
                    err = str(msg.error())
                    log.warning("Kafka error: %s", err)
                    with _lock:
                        _status["last_error"] = err
                    continue

                raw = msg.value()
                received_at = datetime.now(timezone.utc).isoformat()

                try:
                    parsed = _parse_voevent(raw)
                except Exception as exc:
                    log.warning("Parse error on %s: %s", msg.topic(), exc)
                    parsed = {
                        "ivorn": f"unparseable:{received_at}:{msg.offset()}",
                        "role": "",
                        "author": "",
                        "event_time": None,
                        "description": str(exc),
                        "ra": None,
                        "dec": None,
                        "error_radius": None,
                    }

                record = {
                    "topic": msg.topic(),
                    "received_at": received_at,
                    "raw_xml": raw.decode("utf-8", errors="replace"),
                    **parsed,
                }

                if db.insert_event(record):
                    insert_count += 1
                    with _lock:
                        _status["received"] += 1
                        _status["last_event_time"] = received_at

                # prune every 200 new inserts
                if insert_count % 200 == 0 and insert_count > 0:
                    removed = db.prune_old_events(31)
                    if removed:
                        log.info("Pruned %d stale events", removed)

    finally:
        consumer.close()
        with _lock:
            _status["running"] = False
        log.info("GCN consumer stopped")


# ── public API ─────────────────────────────────────────────────────────────────

def start() -> None:
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_loop, daemon=True, name="gcn-consumer")
    _thread.start()


def stop() -> None:
    _stop_event.set()


def get_status() -> dict:
    with _lock:
        return dict(_status)
