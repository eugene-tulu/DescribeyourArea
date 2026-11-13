import json
import os
import tempfile
import unittest
from pathlib import Path

from contract import Connector, ProjectCreate
from projects import create, delete, get, list_projects


class ProjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["PROJECTS_DIR"] = self.tmp.name

    def tearDown(self):
        os.environ.pop("PROJECTS_DIR", None)
        self.tmp.cleanup()

    def _create(self, name="Mara", kind="conservancy"):
        return create(ProjectCreate(
            name=name,
            type=kind,
            geojson={"type": "Point", "coordinates": [35.4, -1.5]},
            connectors=[Connector(
                kind="arcgis",
                name="Survey123 plots",
                config={"url": "https://services.arcgis.com/x/FeatureServer", "layer_id": "0"},
            )],
        ))

    def test_create_get_list_delete(self):
        project = self._create()
        self.assertEqual(len(project.id), 16)
        loaded = get(project.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "Mara")
        self.assertEqual(loaded.type, "conservancy")
        self.assertEqual(loaded.connectors[0].kind, "arcgis")
        self.assertEqual(len(list_projects()), 1)
        self.assertTrue(delete(project.id))
        self.assertIsNone(get(project.id))

    def test_rejects_a_path_traversal_id(self):
        with self.assertRaises(ValueError):
            get("../etc/passwd")

    def test_writes_one_json_per_project(self):
        project = self._create()
        path = Path(self.tmp.name) / f"{project.id}.json"
        self.assertTrue(path.exists())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["name"], "Mara")
        self.assertEqual(data["connectors"][0]["kind"], "arcgis")

    def test_list_skips_a_corrupt_file(self):
        self._create()
        (Path(self.tmp.name) / "deadbeef.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(len(list_projects()), 1)


if __name__ == "__main__":
    unittest.main()