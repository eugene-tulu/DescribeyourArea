"""Tests for the two notification channels.

The split is the point: a webhook is for machines, email is for people, and
neither is required for the sweep to work.
"""

import json
import os
import tempfile
import unittest
import urllib.error
from unittest import mock

import notify


class RenderTests(unittest.TestCase):
    def test_a_breach_names_the_rule_the_value_and_the_source(self):
        subject, body = notify.render({
            "kind": "breach", "rule": "severe-drought", "area": "Melako",
            "value": -38.0, "threshold": -40.0, "month": "2025-12",
            "source": "ERA5", "doi": "10.x", "summary": "A year 40% drier",
        })
        self.assertIn("severe-drought", subject)
        self.assertIn("Melako", subject)
        self.assertIn("-38.0", body)
        self.assertIn("10.x", body)

    def test_a_recovery_says_so_rather_than_repeating_the_breach(self):
        subject, body = notify.render({
            "kind": "recovered", "rule": "drought-watch", "area": "Sera",
            "value": -24.8, "threshold": -20.0, "closed": "2026-03",
        })
        self.assertIn("no longer breaching", body)
        self.assertIn("resolved", subject)

    def test_a_completion_carries_the_thing_you_need_to_fetch_it(self):
        subject, body = notify.render({
            "event": notify.JOB_READY, "indicator": "vegetation_series",
            "area": "Melako", "area_key": "abc123", "months": 189,
            "resolution_m": 231.7, "grid_cells": 42295,
        })
        self.assertIn("ready", subject)
        self.assertIn("189 months", body)
        self.assertIn("231.7", body)
        self.assertIn("abc123", body, "the key is how you retrieve it")

    def test_a_failure_carries_the_reason(self):
        subject, body = notify.render({
            "event": notify.JOB_FAILED, "indicator": "ndvi", "area": "Sera",
            "reason": "no scenes with usable pixels",
        })
        self.assertIn("failed", subject)
        self.assertIn("no scenes", body)

    def test_a_body_never_carries_a_geometry(self):
        for message in (
            {"kind": "breach", "rule": "r", "area": "a", "value": 1, "threshold": 2},
            {"event": notify.JOB_READY, "indicator": "ndvi", "area": "a", "months": 3},
        ):
            subject, body = notify.render(message)
            for banned in ("coordinates", "geometry", "Polygon", "bbox"):
                self.assertNotIn(banned, body)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.previous = {
            k: os.environ.get(k)
            for k in ("RAINFALL_ALERT_WEBHOOK", "AGENTMAIL_API_KEY",
                       "AGENTMAIL_INBOX_ID", "ALERT_EMAIL_TO")
        }
        for key in self.previous:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_nothing_is_configured_by_default(self):
        self.assertFalse(notify.webhook_url())
        self.assertFalse(notify.email_enabled())
        self.assertEqual(notify.recipients(), [])

    def test_email_needs_a_key_an_inbox_and_a_recipient(self):
        os.environ["AGENTMAIL_API_KEY"] = "am_test"
        self.assertFalse(notify.email_enabled(), "a key alone is not enough")
        os.environ["AGENTMAIL_INBOX_ID"] = "inbox"
        self.assertFalse(notify.email_enabled(), "still needs a recipient")
        os.environ["ALERT_EMAIL_TO"] = "a@example.org"
        self.assertTrue(notify.email_enabled())

    def test_recipients_are_split_and_trimmed(self):
        os.environ["ALERT_EMAIL_TO"] = " a@example.org , b@example.org ,, "
        self.assertEqual(notify.recipients(), ["a@example.org", "b@example.org"])

    def test_the_agentmail_endpoint_is_built_from_the_inbox(self):
        os.environ["AGENTMAIL_INBOX_ID"] = "inbox123"
        self.assertEqual(
            notify._agentmail_endpoint(),
            "https://api.agentmail.to/inboxes/inbox123/messages/send",
        )


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.previous = {
            k: os.environ.get(k)
            for k in ("RAINFALL_ALERT_WEBHOOK", "AGENTMAIL_API_KEY",
                       "AGENTMAIL_INBOX_ID", "ALERT_EMAIL_TO")
        }
        for key in self.previous:
            os.environ.pop(key, None)
        self.addCleanup(self._restore)

    def _restore(self):
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_an_unconfigured_webhook_is_a_no_op(self):
        self.assertFalse(notify.send_webhook({"kind": "breach"}))

    def test_unconfigured_email_sends_nothing(self):
        self.assertEqual(notify.send_email({"kind": "breach"}), 0)

    def test_the_webhook_payload_is_slack_shaped(self):
        os.environ["RAINFALL_ALERT_WEBHOOK"] = "https://hooks.example/abc"
        captured = {}

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data)
            captured["headers"] = dict(request.headers)
            return _Response()

        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            self.assertTrue(notify.send_webhook({
                "kind": "breach", "rule": "drought-watch", "area": "Sera",
                "value": -24.8, "threshold": -20.0,
            }))
        # Slack reads text, Discord reads content; sending both means neither is
        # an empty message.
        self.assertIn("text", captured["body"])
        self.assertIn("content", captured["body"])
        self.assertEqual(captured["body"]["rule"], "drought-watch")
        self.assertEqual(captured["body"]["value"], -24.8)

    def test_email_posts_once_per_recipient_with_a_bearer_token(self):
        os.environ.update({
            "AGENTMAIL_API_KEY": "am_secret",
            "AGENTMAIL_INBOX_ID": "inbox123",
            "ALERT_EMAIL_TO": "a@example.org,b@example.org",
        })
        calls = []

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout=None):
            calls.append({
                "url": request.full_url,
                "body": json.loads(request.data),
                "headers": {k.lower(): v for k, v in request.header_items()},
            })
            return _Response()

        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            sent = notify.send_email({
                "event": notify.JOB_READY, "indicator": "vegetation_series",
                "area": "Melako", "area_key": "abc", "months": 189,
            })
        self.assertEqual(sent, 2)
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIn("/inboxes/inbox123/messages/send", call["url"])
            self.assertEqual(call["headers"]["authorization"], "Bearer am_secret")
            self.assertIn("subject", call["body"])
            self.assertIn("text", call["body"])
        self.assertEqual([c["body"]["to"] for c in calls], ["a@example.org", "b@example.org"])

    def test_a_failing_channel_does_not_stop_the_others(self):
        sent = []

        def broken(_message):
            raise urllib.error.URLError("endpoint down")

        def working(message):
            sent.append(message)

        notify.dispatch({"kind": "breach"}, channels=(broken, working))
        self.assertEqual(len(sent), 1, "the healthy channel still ran")

    def test_dispatch_never_raises(self):
        def broken(_message):
            raise urllib.error.URLError("down")

        notify.dispatch({"kind": "breach"}, channels=(broken,))


if __name__ == "__main__":
    unittest.main()
