"""Alerts for precomputed rainfall series.

The retention argument for a monitoring product is that a dashboard is visited
once and a subscription is visited monthly, and the difference is a notification.
So the series is re-checked on every worker sweep and a rule that crosses sends a
message.

Two properties matter more than the threshold itself:

- **Deduplicated.** An alert fires once per episode, not on every sweep. A dry
  season would otherwise produce a notification an hour for months.
- **Self-describing.** Every message carries the value, the baseline it was
  compared against, the window, the source and its DOI, because the recipient
  has to be able to defend the number to somebody.

Delivery is a single generic webhook, which covers Slack, Discord and anything
else that accepts a JSON POST. Unconfigured, alerts are evaluated and recorded
but nothing is sent, so the rules are testable and the failure mode is a log line
rather than a crash.
"""

from __future__ import annotations

import datetime
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

import notify
import rainfall

ALERT_VERSION = 1
STATE_VERSION = 1

# Metric keys a rule may name. Rainfall series carry these under "summary".
METRICS = {
    "trailing_12m_anomaly_pct": lambda s: _get(s, "trailing_12m", "anomaly_pct"),
    "latest_anomaly_pct": lambda s: _get(s, "latest_anomaly_pct"),
    "trailing_3m_anomaly_pct": lambda s: _three_month(s),
}


def _get(summary: dict, *path) -> Optional[float]:
    node: Any = summary
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, (int, float)) else None


def _three_month(summary: dict) -> Optional[float]:
    rows = summary.get("recent_3m") or []
    values = [r.get("anomaly_pct") for r in rows if isinstance(r.get("anomaly_pct"), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def _default_rules() -> list[dict]:
    """Shipped thresholds, from the drought and flood conditions the audience asked
    about. Percentages are against each calendar month's 1991-2020 normal, so they
    mean the same thing in a wet and a dry month."""
    return [
        {
            "id": "severe-drought",
            "metric": "trailing_12m_anomaly_pct",
            "below": -40.0,
            "summary": "A year at least 40% drier than the 1991-2020 normal",
        },
        {
            "id": "drought-watch",
            "metric": "trailing_12m_anomaly_pct",
            "below": -20.0,
            "summary": "A year at least 20% drier than the 1991-2020 normal",
        },
        {
            "id": "flood-watch",
            "metric": "trailing_12m_anomaly_pct",
            "above": 40.0,
            "summary": "A year at least 40% wetter than the 1991-2020 normal",
        },
    ]


def rules() -> list[dict]:
    """Configured rules, or the shipped defaults. ``RAINFALL_ALERT_RULES`` is JSON."""
    raw = (os.getenv("RAINFALL_ALERT_RULES") or "").strip()
    if not raw:
        return _default_rules()
    try:
        parsed = json.loads(raw)
    except ValueError:
        print("RAINFALL_ALERT_RULES is not valid JSON; using the defaults", flush=True)
        return _default_rules()
    if not isinstance(parsed, list) or not parsed:
        return _default_rules()
    return [rule for rule in parsed if isinstance(rule, dict) and rule.get("id")]


def state_path() -> Path:
    return rainfall.cache_dir() / "alert-state.json"


def read_state() -> dict:
    path = state_path()
    if not path.exists():
        return {"v": STATE_VERSION, "episodes": {}}
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"v": STATE_VERSION, "episodes": {}}
    if state.get("v") != STATE_VERSION:
        return {"v": STATE_VERSION, "episodes": {}}
    state.setdefault("episodes", {})
    return state


def write_state(state: dict) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=1, sort_keys=True))
    temporary.replace(path)


def load_series(key: str) -> Optional[dict]:
    payload = rainfall.read_cache(key)
    if not payload or payload.get("status") not in (None, "ok"):
        return None
    return payload if payload.get("series") else None


def evaluate(key: str, payload: dict, configured=None) -> list[dict]:
    """Which rules are currently breached for one area, with the evidence.

    Evaluation is stateless; deduplication happens in :func:`notify` so a breach
    and its first message are separable concerns.
    """
    summary = payload.get("summary") or {}
    breached = []
    for rule in (configured if configured is not None else rules()):
        metric = rule.get("metric")
        reader = METRICS.get(metric)
        if reader is None:
            continue
        value = reader(summary)
        if value is None:
            continue
        below, above = rule.get("below"), rule.get("above")
        hit = (below is not None and value <= below) or (above is not None and value >= above)
        if not hit:
            continue
        breached.append({
            "rule": rule.get("id"),
            "metric": metric,
            "threshold": below if below is not None else above,
            "direction": "below" if below is not None else "above",
            "value": round(float(value), 1),
            "summary": rule.get("summary"),
            "series_months": len(payload.get("series", [])),
            "retrieved": payload.get("retrieved"),
            "source": payload.get("source"),
            "doi": payload.get("doi"),
            "climatology": (payload.get("climatology") or {}).get("standard"),
        })
    return breached


def _delivered(result: Any) -> bool:
    """A sender returns False when nothing was actually delivered.

    Without this the caller cannot tell an alert that reached a webhook from one
    that was swallowed because no webhook is configured, and "what was sent" would
    be a claim rather than a fact.
    """
    return result is not False


def _episode_signature(key: str, rule: str) -> str:
    return f"{key}:{rule}"


def _last_value(payload: dict) -> Optional[str]:
    series = payload.get("series") or []
    return series[-1].get("month") if series else None


def notify(
    key: str,
    payload: dict,
    breached: list[dict],
    state: dict,
    send: Callable[[dict], Any],
) -> list[dict]:
    """Send first-time breaches and close episodes that have recovered.

    Recovery is announced too. A monitoring notification that only ever says
    "bad" teaches people to ignore it; the way back to normal is the signal that
    says the watch is over.
    """
    sent: list[dict] = []
    episodes = state.setdefault("episodes", {})
    latest = _last_value(payload)
    active = {breach["rule"] for breach in breached}

    for signature in list(episodes):
        area_key, rule = signature.rsplit(":", 1)
        if area_key == key and rule not in active:
            closed = episodes.pop(signature)
            message = {
                "v": ALERT_VERSION,
                "kind": "recovered",
                "rule": rule,
                "area_key": key,
                "area": closed.get("area"),
                "opened": closed.get("opened"),
                "closed": latest,
                "value": closed.get("value"),
                "threshold": closed.get("threshold"),
                "summary": f"No longer breaching: {closed.get('summary')}",
                "source": payload.get("source"),
                "doi": payload.get("doi"),
            }
            if _delivered(send(message)):
                sent.append(message)

    for breach in breached:
        signature = _episode_signature(key, breach["rule"])
        if signature in episodes:
            episodes[signature]["last_seen"] = latest
            continue
        message = {
            "v": ALERT_VERSION,
            "kind": "breach",
            "rule": breach["rule"],
            "area_key": key,
            "area": payload.get("label"),
            "value": breach["value"],
            "threshold": breach["threshold"],
            "direction": breach["direction"],
            "summary": breach["summary"],
            "month": latest,
            "climatology": breach["climatology"],
            "source": breach["source"],
            "doi": breach["doi"],
        }
        if _delivered(send(message)):
            sent.append(message)
        episodes[signature] = {
            "area": message["area"],
            "opened": latest,
            "last_seen": latest,
            "value": breach["value"],
            "threshold": breach["threshold"],
            "summary": breach["summary"],
        }
    return sent


def _post_webhook(message: dict, timeout: float = 10.0) -> bool:
    url = (os.getenv("RAINFALL_ALERT_WEBHOOK") or "").strip()
    if not url:
        return False
    body = {
        "text": (
            f"[{message.get('rule')}] {message.get('area') or message.get('area_key')}: "
            f"{message.get('summary')}\n"
            f"value {message.get('value')} (threshold {message.get('threshold')}, "
            f"against {message.get('climatology') or 'the 1991-2020 normal'}), "
            f"month {message.get('month') or message.get('closed')}\n"
            f"source: {message.get('source')}"
            + (f" {message['doi']}" if message.get("doi") else "")
        ),
        "area_key": message.get("area_key"),
        "rule": message.get("rule"),
        "kind": message.get("kind"),
        "value": message.get("value"),
        "threshold": message.get("threshold"),
        "month": message.get("month") or message.get("closed"),
        "source": message.get("source"),
        "doi": message.get("doi"),
    }
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", "user-agent": "GeoContextualize/1.11"},
    )
    with urllib.request.urlopen(request, timeout=timeout):  # noqa: S310 - operator-supplied URL
        pass
    return True


def evaluate_and_notify(keys: list[str], send: Optional[Callable[[dict], None]] = None) -> list[dict]:
    """Evaluate every given area and notify on new breaches or recoveries.

    Returns the messages that were **delivered**, so a caller can report a fact
    rather than a claim. A webhook that is not configured, or that is down, is not
    an error: the episode is still recorded, so configuring a webhook later does
    not re-announce an alert that is already open.
    """
    state = read_state()
    configured = rules()
    delivery = send if send is not None else _post_webhook

    def guarded(message: dict) -> bool:
        try:
            return delivery(message)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # A channel that is down must not stop the sweep, and must not lose the
            # record of which episodes are open.
            print(f"alert delivery failed: {type(exc).__name__}: {exc}", flush=True)
            return False

    sent: list[dict] = []
    before = len(state.get("episodes", {}))
    for key in keys:
        payload = load_series(key)
        if payload is None:
            continue
        sent += notify(key, payload, evaluate(key, payload, configured), state, guarded)
    # Persist whenever an episode opened or closed, not only when something was
    # delivered. Otherwise configuring a webhook later re-announces every open
    # alert, which is precisely the moment a notification should mean something.
    if len(state.get("episodes", {})) != before:
        write_state(state)
    return sent


def open_episodes() -> list[dict]:
    state = read_state()
    return [
        {"signature": signature, **episode}
        for signature, episode in sorted(state.get("episodes", {}).items())
    ]
