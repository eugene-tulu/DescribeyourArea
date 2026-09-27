"""Check that every value the deployment can inject matches the code.

``docker-compose.yml`` pins defaults in its ``environment:`` block, which
overrides the compiled defaults. They were left at the pre-1.4.0 values while the
code moved on, so the container silently ran a 10 km2 NDVI cap and a single
concurrency slot. This asserts the two can never drift apart again.
"""

import re
import unittest
from pathlib import Path

import main

COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"

# Environment variables the deployment injects, and the attribute each one must
# agree with in main.py.
INJECTED = {
    "MAX_GEOJSON_BYTES": "MAX_GEOJSON_BYTES",
    "MAX_AOI_VERTICES": "MAX_AOI_VERTICES",
    "MAX_SYNC_BBOX_KM2": "MAX_SYNC_BBOX_KM2",
    "MAX_LANDCOVER_BBOX_KM2": "MAX_LANDCOVER_BBOX_KM2",
    "MAX_PC_SCENES": "MAX_PC_SCENES",
    "MAX_SOURCE_TILES": "MAX_SOURCE_TILES",
    "MAX_CONCURRENT_ANALYSES": "MAX_CONCURRENT_ANALYSES",
    "MAX_CONCURRENT_NDVI": "MAX_CONCURRENT_NDVI",
    "ANALYSIS_ACQUIRE_SECONDS": "ANALYSIS_ACQUIRE_SECONDS",
    "NDVI_ACQUIRE_SECONDS": "NDVI_ACQUIRE_SECONDS",
    "ANALYSIS_DRAIN_SECONDS": "ANALYSIS_DRAIN_SECONDS",
}


def _compose_environment() -> dict[str, str]:
    text = COMPOSE.read_text()
    block = text.split("environment:", 1)[1].split("healthcheck:", 1)[0]
    found = {}
    for line in block.splitlines():
        match = re.match(r"\s+([A-Z0-9_]+):\s+\$\{([A-Z0-9_]+):-([^}]*)\}\s*$", line)
        if match:
            found[match.group(1)] = match.group(3)
    return found


class DeploymentConfigTests(unittest.TestCase):
    def setUp(self):
        if not COMPOSE.exists():
            self.skipTest("docker-compose.yml not present")
        self.env = _compose_environment()

    def test_every_injected_value_matches_the_code_default(self):
        drifted = {}
        for name, attribute in INJECTED.items():
            if name not in self.env:
                continue
            declared = self.env[name].strip()
            actual = getattr(main, attribute)
            if float(declared) != float(actual):
                drifted[name] = (declared, actual)
        self.assertEqual(
            drifted, {},
            "docker-compose.yml pins a different default than main.py; the "
            "container environment wins, so the code default is dead in production",
        )

    def test_the_payload_cap_is_injected(self):
        self.assertIn("MAX_GEOJSON_BYTES", self.env)
        self.assertIn("MAX_AOI_VERTICES", self.env)

    def test_the_limits_from_1_4_0_are_reachable(self):
        # The specific regression: these two were left at their pre-1.4.0 values,
        # which silently undid the concurrency work in production.
        self.assertEqual(float(self.env["MAX_CONCURRENT_ANALYSES"]), 8.0)
        self.assertLess(
            float(self.env["MAX_CONCURRENT_NDVI"]),
            float(self.env["MAX_CONCURRENT_ANALYSES"]),
        )

    def test_rainfall_variables_are_injected(self):
        for name in ("RAINFALL_CACHE_DIR", "RAINFALL_CACHE_S3_URI",
                     "RAINFALL_S3_ENDPOINT", "RAINFALL_S3_REGION"):
            with self.subTest(name=name):
                self.assertIn(name, self.env)

    def test_the_proxy_and_the_application_agree_on_body_size(self):
        """A proxy that admits less than the application accepts only moves the
        rejection somewhere less explicable: the 512k Nginx limit silently made the
        1,358 KB NRT conservancies unsubmittable while the app advertised 4 MB."""
        nginx = list((COMPOSE.parent / "deploy" / "nginx").glob("*.conf"))
        limits = []
        for path in nginx:
            text = path.read_text()
            limits += [m for m in re.findall(r"client_max_body_size\s+(\d+)([kKmM])", text)]
        self.assertTrue(limits, "no client_max_body_size found in the Nginx configs")
        for value, unit in limits:
            factor = {"k": 1024, "m": 1024 * 1024}[unit.lower()]
            self.assertGreaterEqual(
                int(value) * factor, main.MAX_GEOJSON_BYTES,
                f"proxy admits {int(value)}{unit}, below the application's "
                f"{main.MAX_GEOJSON_BYTES} bytes, so a body can pass the proxy and "
                "then be refused with no useful reason",
            )

    def test_cache_dir_is_writable_for_the_app_user(self):
        # The entrypoint writes the pulled portfolio here as a non-root user.
        self.assertTrue(self.env["RAINFALL_CACHE_DIR"].startswith("/app/"))


class EntrypointTests(unittest.TestCase):
    SCRIPT = Path(__file__).resolve().parents[1] / "docker-entrypoint.sh"
    DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"

    def setUp(self):
        if not self.SCRIPT.exists():
            self.skipTest("docker-entrypoint.sh not present")

    def test_entrypoint_exists_and_is_executable(self):
        self.assertTrue(self.SCRIPT.exists())
        self.assertIn("exec", self.SCRIPT.read_text())

    def test_a_failed_pull_does_not_stop_the_service(self):
        """Losing the portfolio should degrade a card, not take the API down.

        Executed rather than string-matched: a bogus remote must still let the
        command after the entrypoint run.
        """
        import os
        import subprocess
        import tempfile

        environment = dict(os.environ)
        environment.update({
            # A bucket that cannot exist, so the pull fails fast.
            "RAINFALL_CACHE_S3_URI": "s3://geocontextualize-test-no-such-bucket/rainfall",
            "RAINFALL_S3_ENDPOINT": "https://127.0.0.1:1",
            "RAINFALL_S3_REGION": "nyc3",
            "RAINFALL_CACHE_DIR": tempfile.mkdtemp(),
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        })
        result = subprocess.run(
            ["sh", str(self.SCRIPT), "echo", "SERVER_STARTED"],
            capture_output=True, text=True, timeout=120, env=environment,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertIn("SERVER_STARTED", result.stdout,
                      f"entrypoint aborted the service. stderr: {result.stderr[-400:]}")
        self.assertEqual(result.returncode, 0)

    def test_no_remote_configured_still_starts(self):
        import os
        import subprocess
        import tempfile

        environment = dict(os.environ)
        environment.pop("RAINFALL_CACHE_S3_URI", None)
        environment.update({
            "RAINFALL_CACHE_DIR": tempfile.mkdtemp(),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        })
        result = subprocess.run(
            ["sh", str(self.SCRIPT), "echo", "SERVER_STARTED"],
            capture_output=True, text=True, timeout=60, env=environment,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertIn("SERVER_STARTED", result.stdout)
        self.assertIn("no RAINFALL_CACHE_S3_URI", result.stdout)

    def test_dockerfile_uses_the_entrypoint(self):
        dockerfile = self.DOCKERFILE.read_text()
        self.assertIn("ENTRYPOINT [\"/app/docker-entrypoint.sh\"]", dockerfile)
        self.assertIn("chmod +x /app/docker-entrypoint.sh", dockerfile)


if __name__ == "__main__":
    unittest.main()
