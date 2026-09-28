"""Notifications, on two channels for two audiences.

**Webhook is for machines.** The alerting pipeline runs on it, because a webhook
delivers in seconds and returns something an automated system can act on.

**Email is for people.** A breach is a thing a conservancy manager should hear
about, and so is "the thing you asked us to compute is ready" -- the case where
someone submitted an area and then went away, which a polling browser tab does not
solve. Email is the wrong transport for the pipeline, so it never carries it.

Neither is required. Unconfigured, every sender is a no-op and the sweep is
unaffected, which is what makes them safe to add.

A note on what an email may contain: the area *label* and its cache key, the
value, the threshold and the source. Never a submitted geometry, never a raw area,
never a client address -- the same restraint the usage log follows, because an
alert that names a place is a disclosure of interest in that place.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

DEFAULT_TIMEOUT = 10.0
USER_AGENT = "GeoContextualize/1.11"

# Triggers. A breach is a claim about a place; a completion is a claim about
# work, and conflating them makes both harder to trust.
ALERT_BREACH = "alert.breach"
ALERT_RECOVERY = "alert.recovery"
JOB_READY = "job.ready"
JOB_FAILED = "job.failed"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def webhook_url() -> str:
    return (os.getenv("RAINFALL_ALERT_WEBHOOK") or "").strip()


def email_enabled() -> bool:
    return bool(
        (os.getenv("AGENTMAIL_API_KEY") or "").strip()
        and (os.getenv("AGENTMAIL_INBOX_ID") or "").strip()
        and recipients()
    )


def recipients() -> list[str]:
    raw = (os.getenv("ALERT_EMAIL_TO") or "")
    return [address.strip() for address in raw.split(",") if address.strip()]


def _agentmail_endpoint() -> str:
    inbox = (os.getenv("AGENTMAIL_INBOX_ID") or "").strip()
    return f"https://api.agentmail.to/inboxes/{inbox}/messages/send"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render(message: dict) -> tuple[str, str]:
    """Return ``(subject, body)`` for a message, for either channel."""
    kind = message.get("kind")
    if kind == "recovered" or message.get("event") == ALERT_RECOVERY:
        subject = f"[resolved] {message.get('rule')}: {message.get('area') or message.get('area_key')}"
        body = (
            f"{message.get('area') or message.get('area_key')} is no longer breaching.\n"
            f"Rule: {message.get('rule')}\n"
            f"Was {message.get('value')} against a threshold of {message.get('threshold')}\n"
            f"Closed at {message.get('closed')}\n"
            f"Source: {message.get('source')}"
        )
    elif message.get("event") in (JOB_READY, JOB_FAILED):
        ready = message.get("event") == JOB_READY
        subject = (
            f"[{'ready' if ready else 'failed'}] {message.get('indicator')} "
            f"for {message.get('area') or message.get('area_key')}"
        )
        detail = (
            f"{message.get('months')} months, "
            f"{message.get('resolution_m')} m, {message.get('grid_cells')} cells in area"
            if message.get("months")
            else message.get("reason") or "no detail"
        )
        body = (
            f"{message.get('area') or message.get('area_key')}\n"
            f"Indicator: {message.get('indicator')}\n"
            f"{detail}\n"
            # The key is the retrieval handle, so it is in the body even when a
            # label is present. Without it the message is not actionable.
            f"Cache key: {message.get('area_key')}\n"
            f"{message.get('retrieve_hint') or 'POST /rainfall'}"
        )
    else:
        subject = f"[{message.get('rule')}] {message.get('area') or message.get('area_key')}"
        body = (
            f"{message.get('summary') or 'threshold crossed'}\n"
            f"Value {message.get('value')} against a threshold of {message.get('threshold')}, "
            f"versus {message.get('climatology') or 'the climatological normal'}\n"
            f"Month {message.get('month')}\n"
            f"Source: {message.get('source')}"
            + (f" {message['doi']}" if message.get("doi") else "")
        )
    return subject, body


# --------------------------------------------------------------------------
# Transports
# --------------------------------------------------------------------------

def send_webhook(message: dict, timeout: float = DEFAULT_TIMEOUT) -> bool:
    """POST the message as JSON. The payload is Slack-shaped, so a Slack,
    Mattermost, n8n or Zapier endpoint accepts it unchanged. Discord wants
    ``content`` rather than ``text`` and is the one common exception."""
    url = webhook_url()
    if not url:
        return False
    subject, body = render(message)
    payload = {
        "text": f"*{subject}*\n{body}",
        "content": f"*{subject}*\n{body}",
        "area_key": message.get("area_key"),
        "rule": message.get("rule"),
        "kind": message.get("kind"),
        "event": message.get("event"),
        "value": message.get("value"),
        "threshold": message.get("threshold"),
        "month": message.get("month") or message.get("closed"),
        "source": message.get("source"),
        "doi": message.get("doi"),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", "user-agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout):  # noqa: S310 - operator-supplied URL
        pass
    return True


def send_email(message: dict, timeout: float = DEFAULT_TIMEOUT) -> int:
    """Send to every configured recipient through AgentMail. Returns the count."""
    if not email_enabled():
        return 0
    subject, body = render(message)
    sent = 0
    for address in recipients():
        payload = {"to": address, "subject": subject, "text": body}
        request = urllib.request.Request(
            _agentmail_endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "authorization": f"Bearer {(os.getenv('AGENTMAIL_API_KEY') or '').strip()}",
                "content-type": "application/json",
                "user-agent": USER_AGENT,
            },
        )
        # urllib raises on a non-2xx, including the 429 the API documents, so the
        # caller's retry policy sees the failure rather than a silent success.
        with urllib.request.urlopen(request, timeout=timeout):  # noqa: S310 - fixed host
            pass
        sent += 1
    return sent


def dispatch(
    message: dict,
    channels: Optional[tuple[Callable[..., Any], ...]] = None,
) -> dict[str, Any]:
    """Send on every configured channel, reporting each outcome separately.

    A channel that fails never prevents another from being tried, and never
    propagates: a notification service being down must not stop the sweep or
    lose the record of which episodes are open.
    """
    for channel in channels or (send_webhook, send_email):
        name = getattr(channel, "__name__", str(channel))
        try:
            channel(message)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"notification via {name} failed: {type(exc).__name__}: {exc}", flush=True)
