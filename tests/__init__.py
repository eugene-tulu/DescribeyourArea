"""Test package.

Disables .env loading before the application is imported, so a developer's own
bucket and credentials never enter a test process. Without this, an assertion
about a cache miss can be satisfied by a live remote and a run mutates real
storage.
"""

import os

os.environ["GEOCONTEXT_NO_DOTENV"] = "1"
