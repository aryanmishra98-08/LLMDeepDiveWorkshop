"""
Voice Agent Server (Ultravox):
Keeps ULTRAVOX_API_KEY on the server side — never exposed to the browser.
Exposes POST /start-call which returns a { joinUrl } the browser SDK uses to connect.
"""

import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
import requests
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# Load credentials from keys/.env
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / "keys" / ".env"
if _ENV_PATH.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)

# ── Logging setup ─────────────────────────────────────────────────────────────
_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_RUN_TS = time.strftime("%Y%m%d_%H%M%S")
_LOG_FILE = _LOGS_DIR / f"voice_{_RUN_TS}.log"

log = logging.getLogger("voice_agent_server")
log.setLevel(logging.DEBUG)
log.handlers.clear()
log.propagate = False

_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_formatter)

_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_formatter)

log.addHandler(_console_handler)
log.addHandler(_file_handler)
log.info("Logging initialized | file: %s", _LOG_FILE)

# ── Config ────────────────────────────────────────────────────────────────────

ULTRAVOX_API_KEY = os.getenv("ULTRAVOX_API_KEY")
ULTRAVOX_API_URL = "https://api.ultravox.ai/api/calls"
ULTRAVOX_MODEL = os.getenv("ULTRAVOX_MODEL", "ultravox-v0.7")
APP_URL = os.getenv("APP_URL", "http://127.0.0.1:5000")
AUTO_OPEN_BROWSER = os.getenv("AUTO_OPEN_BROWSER", "true").strip().lower() in {"1", "true", "yes", "on"}

_BROWSER_LOCK = threading.Lock()
_BROWSER_OPEN_SCHEDULED = False

# Nova's system prompt — matches the slides exactly
SYSTEM_PROMPT = """You are Nova, a warm and professional receptionist for BrightStart Coaching — a career coaching company that helps professionals navigate career transitions and grow into leadership roles.

Your job is to:
- Answer questions about BrightStart's coaching services
- Understand what the caller needs and what they're hoping to achieve
- Offer to book a free 30-minute intro call if they seem interested

Keep your responses concise — 2 to 3 sentences maximum. Be human. Don't sound scripted. If you don't know something specific about BrightStart's offerings, say so honestly and offer to find out or connect them with the right person.

Do not make up pricing, session counts, or specific coach names. When in doubt, offer the intro call.

When the conversation naturally concludes — the caller says goodbye, thanks you, or indicates they're done — use the hangUp tool to end the call politely."""

AGENT_CONFIG = {
    "systemPrompt":        SYSTEM_PROMPT,
    "model":               ULTRAVOX_MODEL,
    "voice":               "Jessica",  # Ultravox voice ID — see ultravox.ai/docs for alternatives
    "temperature":         0.4,         # Low-ish temperature keeps answers consistent but not robotic
    "selectedTools": [
        {
            "toolName": "hangUp",
        }
    ],
    "firstSpeakerSettings": {
        "agent": {
            # uninterruptible=True prevents the user cutting off Nova's opening line,
            # giving the greeting a chance to land cleanly before the conversation begins.
            "uninterruptible": True,
            "text": "Hi, this is Nova from BrightStart Coaching. How can I help you today?",
            "delay": "2s"
        }
    },
    "initialOutputMedium": "MESSAGE_MEDIUM_VOICE",
    "joinTimeout":         "5s",    # seconds the browser has to join before the call is abandoned
    "maxDuration":         "300s",  # hard cap of 5 minutes per call
    "timeExceededMessage": "We've reached our 5-minute call limit. I can help you book a quick follow-up to continue.",
    "vadSettings": {
        # VAD (Voice Activity Detection) controls when Nova treats a pause as end-of-turn.
        # turnEndpointDelay: silence after the user stops speaking before Nova responds (0.35 s).
        # minimumTurnDuration: shortest speech burst treated as a real utterance (0.45 s).
        # minimumInterruptionDuration: shortest burst that counts as an interruption (0.25 s).
        "turnEndpointDelay":         "0.35s",
        "minimumTurnDuration":       "0.45s",
        "minimumInterruptionDuration": "0.25s",
    },
}

# ── App ───────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=".")
CORS(app)  # Allow browser to call this server


def _open_in_preferred_browser(url: str) -> None:
    """Open *url* in Google Chrome on macOS when available, otherwise use the system default.

    Chrome is preferred because it reliably grants microphone access to localhost
    without extra permission prompts on macOS. The function falls back gracefully
    to ``webbrowser.open_new_tab`` on other platforms or when Chrome is absent.

    Args:
        url: The fully-qualified URL to open (e.g. ``http://127.0.0.1:5000``).
    """
    if sys.platform == "darwin":
        chrome_app = "/Applications/Google Chrome.app"
        if Path(chrome_app).exists():
            result = subprocess.run(
                ["open", "-a", "Google Chrome", url],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                log.info("Opened app in Google Chrome | url: %s", url)
                return
            log.warning("Chrome launch failed, falling back to default browser | stderr: %s", result.stderr.strip())

    opened = webbrowser.open_new_tab(url)
    if opened:
        log.info("Opened app in default browser | url: %s", url)
    else:
        log.warning("Could not auto-open browser. Please open manually: %s", url)


def schedule_browser_launch(url: str, delay_seconds: float = 0.8) -> None:
    """Schedule a one-time browser launch in a background daemon thread.

    The ``_BROWSER_OPEN_SCHEDULED`` flag + ``_BROWSER_LOCK`` guard ensures this
    launches at most once per server process, even if the function is called
    multiple times (e.g. from both the startup banner and a reload). The daemon
    thread is intentionally non-blocking so Flask startup is not delayed.

    Args:
        url: The URL to open in the browser.
        delay_seconds: How long to wait after scheduling before actually opening
                       the browser. Defaults to 0.8 s to allow Flask to finish
                       binding to the port before the browser hits the server.
    """
    global _BROWSER_OPEN_SCHEDULED
    if not AUTO_OPEN_BROWSER:
        log.info("AUTO_OPEN_BROWSER disabled; skipping browser launch")
        return

    with _BROWSER_LOCK:
        if _BROWSER_OPEN_SCHEDULED:
            return
        _BROWSER_OPEN_SCHEDULED = True

    def _launch() -> None:
        time.sleep(delay_seconds)
        try:
            _open_in_preferred_browser(url)
        except Exception as exc:  # pragma: no cover
            log.warning("Auto-open browser failed: %s", exc)

    threading.Thread(target=_launch, daemon=True, name="browser-launch").start()

# ── Preflight ─────────────────────────────────────────────────────────────────

if not ULTRAVOX_API_KEY:
    log.error("ULTRAVOX_API_KEY is not set — add it to keys/.env")
    sys.exit(1)
# Log the full agent config at DEBUG level so it's captured in the log file
# without cluttering the INFO-level console output.
log.debug("Agent config: model=%s | voice=%s | maxDuration=%s | joinTimeout=%s",
          AGENT_CONFIG["model"], AGENT_CONFIG["voice"],
          AGENT_CONFIG["maxDuration"], AGENT_CONFIG["joinTimeout"])
# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "nova_demo.html")


@app.route("/start-call", methods=["POST"])
def start_call():
    """Create a new Ultravox call and return the joinUrl to the browser.

    The browser calls this endpoint (``POST /start-call``) to initiate a voice
    session.  The server forwards ``AGENT_CONFIG`` to the Ultravox API using the
    server-side ``ULTRAVOX_API_KEY``, which is never exposed to the client.

    Returns:
        JSON ``{"joinUrl": "<wss://...>"}`` on success (HTTP 200).
        JSON ``{"error": "<message>"}`` on failure with the upstream HTTP status
        code, or 502 for connection-level errors.
    """
    try:
        _t0 = time.perf_counter()
        log.info("Creating Ultravox call | model: %s | voice: %s", AGENT_CONFIG["model"], AGENT_CONFIG["voice"])
        response = requests.post(
            ULTRAVOX_API_URL,
            headers={
                "X-API-Key":    ULTRAVOX_API_KEY,
                "Content-Type": "application/json",
            },
            json=AGENT_CONFIG,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        join_url = data.get("joinUrl")
        if not join_url:
            return jsonify({"error": "No joinUrl returned from Ultravox API"}), 500

        elapsed = time.perf_counter() - _t0
        log.info("Call created | elapsed: %.2fs | joinUrl: %s…", elapsed, join_url[:60])
        return jsonify({"joinUrl": join_url})

    except requests.exceptions.HTTPError as e:
        response = e.response
        status_code = response.status_code if response is not None else 502
        error_body = (response.text if response is not None else str(e))[:500]  # truncate large bodies
        # Log a targeted message for common failure modes so they're easy to diagnose.
        if status_code == 401:
            log.error("Ultravox authentication failed (401) — check ULTRAVOX_API_KEY in keys/.env")
        elif status_code == 429:
            log.error("Ultravox rate limit reached (429) — wait a moment and retry")
        elif status_code >= 500:
            log.error("Ultravox server error (%s) — try again shortly", status_code)
        else:
            log.error("Ultravox API error: %s — %s", status_code, error_body)
        return jsonify({"error": f"Ultravox API error: {error_body}"}), status_code
    except requests.exceptions.RequestException as e:
        log.error("Ultravox request failed: %s", e)
        return jsonify({"error": f"Ultravox request failed: {e}"}), 502


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VOICE AGENT SERVER — Nova (BrightStart Coaching)")
    print("=" * 60)
    print(f"\n  Opening {APP_URL} automatically...")
    print(f"  If it does not open, navigate to {APP_URL}.")
    print("─" * 60 + "\n")

    schedule_browser_launch(APP_URL)
    app.run(debug=False, port=5000)
