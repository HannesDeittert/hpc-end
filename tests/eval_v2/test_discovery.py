from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from steve_recommender.eval_v2.discovery import FileBasedAnatomyDiscovery
from steve_recommender.eval_v2.models import AorticArchAnatomy


class FileBasedAnatomyDiscoveryTests(unittest.TestCase):
    def _write_registry(self, root: Path) -> Path:
        registry_root = root / "anatomy_registry"
        anatomies_root = registry_root / "anatomies"
        tree_a_root = anatomies_root / "Tree_00"
        tree_b_root = anatomies_root / "Tree_01"
        (tree_a_root / "mesh").mkdir(parents=True)
        (tree_b_root / "mesh").mkdir(parents=True)

        (tree_a_root / "mesh" / "simulationmesh.obj").write_text("o sim_a\n", encoding="utf-8")
        (tree_a_root / "mesh" / "visumesh.obj").write_text("o vis_a\n", encoding="utf-8")
        (tree_b_root / "mesh" / "simulationmesh.obj").write_text("o sim_b\n", encoding="utf-8")
        (tree_b_root / "mesh" / "visumesh.obj").write_text("o vis_b\n", encoding="utf-8")

        tree_a_payload = {
            "anatomy_type": "aortic_arch",
            "arch_type": "II",
            "seed": 42,
            "rotation_yzx_deg": [1.0, 2.0, 3.0],
            "scaling_xyzd": [0.9, 1.0, 1.1, 1.2],
            "omit_axis": "z",
            "simulation_mesh_path": "mesh/simulationmesh.obj",
            "visualization_mesh_path": "mesh/visumesh.obj",
            "centerline_bundle_path": None,
        }
        tree_b_payload = {
            "anatomy_type": "aortic_arch",
            "arch_type": "I",
            "seed": 7,
            "rotation_yzx_deg": None,
            "scaling_xyzd": None,
            "omit_axis": None,
            "simulation_mesh_path": "mesh/simulationmesh.obj",
            "visualization_mesh_path": "mesh/visumesh.obj",
            "centerline_bundle_path": None,
        }
        (tree_a_root / "description.json").write_text(
            json.dumps(tree_a_payload, indent=2),
            encoding="utf-8",
        )
        (tree_b_root / "description.json").write_text(
            json.dumps(tree_b_payload, indent=2),
            encoding="utf-8",
        )

        index_payload = {
            "version": 1,
            "anatomies": [
                {
                    "record_id": "Tree_00",
                    "created_at": "2026-04-20T12:00:00",
                    "description_path": "anatomies/Tree_00/description.json",
                },
                {
                    "record_id": "Tree_01",
                    "created_at": "2026-04-20T12:05:00",
                    "description_path": "anatomies/Tree_01/description.json",
                },
            ],
        }
        registry_path = registry_root / "index.json"
        registry_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
        return registry_path

    def test_list_anatomies_reads_directory_registry_and_maps_domain_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_registry(Path(tmp))
            discovery = FileBasedAnatomyDiscovery(registry_path=registry_path)

            anatomies = discovery.list_anatomies()

        self.assertIsInstance(anatomies, tuple)
        self.assertEqual(len(anatomies), 2)
        self.assertTrue(all(isinstance(item, AorticArchAnatomy) for item in anatomies))

        anatomy_a = anatomies[0]
        self.assertEqual(anatomy_a.record_id, "Tree_00")
        self.assertEqual(anatomy_a.anatomy_type, "aortic_arch")
        self.assertEqual(anatomy_a.arch_type, "II")
        self.assertEqual(anatomy_a.seed, 42)
        self.assertEqual(anatomy_a.rotation_yzx_deg, (1.0, 2.0, 3.0))
        self.assertEqual(anatomy_a.scaling_xyzd, (0.9, 1.0, 1.1, 1.2))
        self.assertEqual(anatomy_a.omit_axis, "z")
        self.assertEqual(anatomy_a.created_at, "2026-04-20T12:00:00")
        self.assertIsNone(anatomy_a.centerline_bundle_path)
        self.assertEqual(
            anatomy_a.simulation_mesh_path,
            registry_path.parent / "anatomies" / "Tree_00" / "mesh" / "simulationmesh.obj",
        )
        self.assertEqual(
            anatomy_a.visualization_mesh_path,
            registry_path.parent / "anatomies" / "Tree_00" / "mesh" / "visumesh.obj",
        )

        anatomy_b = anatomies[1]
        self.assertEqual(anatomy_b.record_id, "Tree_01")
        self.assertIsNone(anatomy_b.rotation_yzx_deg)
        self.assertIsNone(anatomy_b.scaling_xyzd)
        self.assertIsNone(anatomy_b.centerline_bundle_path)
        self.assertEqual(
            anatomy_b.simulation_mesh_path,
            registry_path.parent / "anatomies" / "Tree_01" / "mesh" / "simulationmesh.obj",
        )
        self.assertEqual(
            anatomy_b.visualization_mesh_path,
            registry_path.parent / "anatomies" / "Tree_01" / "mesh" / "visumesh.obj",
        )

    def test_get_anatomy_raises_key_error_for_unknown_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_registry(Path(tmp))
            discovery = FileBasedAnatomyDiscovery(registry_path=registry_path)

            with self.assertRaises(KeyError):
                discovery.get_anatomy(record_id="Tree_missing")


if __name__ == "__main__":
    unittest.main()
