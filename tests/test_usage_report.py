"""The usage report is the only way anyone reads the instrumentation.

`summarise` was already tested. What was untested is the part that made the data
worth reading at all: per-dataset detail, a per-day timeline, and the fact that
the report can be built from a log stream full of things that are not events.
"""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

import usage


def _event(**overrides):
    event = {
        "v": usage.EVENT_VERSION,
        "at": "2026-09-28T12:18:39+00:00",
        "aoi_area_km2_band": "0-10",
        "client_prefix": "203.0.113.0/24",
        "datasets_requested": ["dem", "ndvi"],
        "duration_ms": {"dem": 56, "ndvi": 1919},
        "outcomes": {"dem": "ok", "ndvi": "ok"},
        "sensor": "landsat",
        "total_ms": 2423,
    }
    event.update(overrides)
    return event


class BreakdownTests(unittest.TestCase):
    def test_it_counts_which_datasets_were_asked_for(self):
        detail = usage.breakdown([_event(), _event(datasets_requested=["dem"])])
        self.assertEqual(detail["requested"], {"dem": 2, "ndvi": 1})

    def test_it_keeps_latency_per_dataset(self):
        # The total averages over datasets the user may not have wanted, so a
        # good total can hide a slow one. Per-dataset p95 is the only way to see
        # that, and it is the number that decides whether a dataset is worth
        # keeping.
        detail = usage.breakdown([
            _event(duration_ms={"ndvi": 100}, total_ms=100),
            _event(duration_ms={"ndvi": 9000}, total_ms=9000),
        ])
        self.assertEqual(detail["duration_ms"]["ndvi"]["n"], 2)
        self.assertEqual(detail["duration_ms"]["ndvi"]["p95"], 9000)

    def test_it_splits_the_timeline_by_day(self):
        detail = usage.breakdown([
            _event(at="2026-09-27T23:59:59+00:00"),
            _event(at="2026-09-28T00:00:01+00:00"),
        ])
        self.assertEqual(detail["by_day"], {"2026-09-27": 1, "2026-09-28": 1})

    def test_a_dataset_can_fail_without_losing_its_timing(self):
        detail = usage.breakdown([_event(
            outcomes={"ndvi": "failed"}, duration_ms={"ndvi": 30_000})])
        self.assertEqual(detail["outcomes"]["ndvi"], {"failed": 1})
        self.assertEqual(detail["duration_ms"]["ndvi"]["p50"], 30_000)


class RenderTests(unittest.TestCase):
    def test_it_reads_as_a_report_rather_than_json(self):
        text = usage.render([_event()])
        for heading in ("outcomes", "datasets requested", "per dataset",
                        "by day", "client prefixes"):
            self.assertIn(heading, text)
        self.assertNotIn("{", text.split("per dataset")[0])

    def test_it_explains_itself_when_there_is_nothing_to_show(self):
        # Silence is the failure mode worth designing against: a report that
        # prints nothing reads as "no traffic" when the truth is "no data was
        # retained", and those two conclusions lead to opposite decisions.
        text = usage.render([])
        self.assertIn("no usage events found", text)
        self.assertIn("USAGE_EVENTS_PATH", text)

    def test_it_names_a_dataset_the_user_never_requested(self):
        # outcomes can carry a dataset the caller did not ask for, and a report
        # that silently dropped it would hide a server-side surprise.
        text = usage.render([_event(datasets_requested=["dem"],
                                    outcomes={"dem": "ok", "ndvi": "skipped"},
                                    duration_ms={"dem": 10})])
        self.assertIn("ndvi", text)


class CliTests(unittest.TestCase):
    def _run(self, argv, stdin=""):
        buffer = io.StringIO()
        saved, sys.stdin = sys.stdin, io.StringIO(stdin)
        try:
            with redirect_stdout(buffer):
                code = usage.main(argv)
        finally:
            sys.stdin = saved
        return code, buffer.getvalue()

    def test_it_reads_a_log_full_of_things_that_are_not_events(self):
        # A container log is mostly tracebacks and uvicorn lines. Analysing it
        # must not require pre-filtering, or the first thing anyone does is
        # `docker logs | grep` and the tool is one step too many.
        log = (
            "INFO:     Started server process\n"
            + json.dumps(_event()) + "\n"
            + "Traceback (most recent call last):\n"
            + "  File \"main.py\", line 1\n"
            + '{"v": 1, "broken json\n'
            + json.dumps(_event(at="2026-09-27T09:00:00+00:00")) + "\n"
        )
        code, out = self._run(["-"], stdin=log)
        self.assertEqual(code, 0)
        self.assertIn("2 event(s)", out)

    def test_json_mode_emits_both_aggregates(self):
        code, out = self._run(["--json"], stdin=json.dumps(_event()) + "\n")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["summary"]["events"], 1)
        self.assertEqual(payload["detail"]["requested"], {"dem": 1, "ndvi": 1})

    def test_a_missing_file_is_an_error_rather_than_an_empty_report(self):
        code, out = self._run(["/nonexistent/usage-events.jsonl"])
        self.assertEqual(code, 1)

    def test_it_reads_a_pipe_even_without_an_explicit_dash(self):
        # `docker logs <c> | python -m usage` is what anyone will actually type.
        # If that silently reports zero events it is worse than no tool, because
        # zero events and no data look the same and lead to opposite decisions.
        saved, sys.stdin = sys.stdin, _NonTTY(json.dumps(_event()) + "\n")
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = usage.main(["--json"])
        finally:
            sys.stdin = saved
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(buffer.getvalue())["summary"]["events"], 1)


class _NonTTY(io.StringIO):
    """A stdin that claims to be a pipe, so the fallback path is exercised."""

    def isatty(self):
        return False


class SourcePrecedenceTests(unittest.TestCase):
    def test_a_configured_path_beats_a_non_tty_stdin(self):
        # docker compose exec gives a non-TTY stdin, so treating "not a tty" as
        # "a log is being piped in" made the tool read an empty pipe and report
        # no events while the file it was sent to read held two. A wrong source
        # that returns a confident zero is worse than one that errors.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(json.dumps(_event()) + "\n", encoding="utf-8")
            saved_path = os.environ.get("USAGE_EVENTS_PATH")
            saved_stdin = sys.stdin
            os.environ["USAGE_EVENTS_PATH"] = str(path)
            sys.stdin = _NonTTY("")
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    code = usage.main([])
            finally:
                sys.stdin = saved_stdin
                if saved_path is None:
                    os.environ.pop("USAGE_EVENTS_PATH", None)
                else:
                    os.environ["USAGE_EVENTS_PATH"] = saved_path
        self.assertEqual(code, 0)
        self.assertIn("1 event(s)", buffer.getvalue())
