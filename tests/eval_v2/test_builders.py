from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from third_party.stEVE.eve.intervention.fluoroscopy import TrackingOnly
from third_party.stEVE.eve.intervention.simulation import SimulationDummy
from third_party.stEVE.eve.intervention.target import BranchEnd, BranchIndex, Manual
from third_party.stEVE.eve.intervention.vesseltree import AorticArch, ArchType

from steve_recommender.eval_v2.models import (
    AorticArchAnatomy,
    BranchEndTarget,
    BranchIndexTarget,
    FluoroscopySpec,
    ManualTarget,
)

from steve_recommender.eval_v2.builders import (
    build_aortic_arch,
    build_fluoroscopy,
    build_target,
)


class RuntimeBuilderIntegrationTests(unittest.TestCase):
    def _anatomy(self) -> AorticArchAnatomy:
        return AorticArchAnatomy(
            arch_type="II",
            seed=42,
            rotation_yzx_deg=(1.0, 2.0, 3.0),
            scaling_xyzd=(1.0, 1.1, 1.2, 1.3),
            omit_axis="z",
        )

    def test_build_aortic_arch_returns_real_steve_object_and_initializes(self) -> None:
        anatomy = self._anatomy()

        vessel_tree = build_aortic_arch(anatomy)
        vessel_tree.reset()

        self.assertIsInstance(vessel_tree, AorticArch)
        self.assertEqual(vessel_tree.arch_type, ArchType.II)
        self.assertEqual(vessel_tree.seed, 42)
        self.assertEqual(vessel_tree.rotation_yzx_deg, (1.0, 2.0, 3.0))
        self.assertEqual(vessel_tree.scaling_xyzd, (1.0, 1.1, 1.2, 1.3))
        self.assertEqual(vessel_tree.omit_axis, "z")
        self.assertIsNotNone(vessel_tree.branches)
        self.assertGreater(len(vessel_tree.branches), 0)
        self.assertIsNotNone(vessel_tree.insertion)
        self.assertIsNotNone(vessel_tree.centerline_coordinates)
        self.assertGreater(vessel_tree.centerline_coordinates.shape[0], 0)

    def test_build_aortic_arch_attaches_visualization_mesh_path_when_present(self) -> None:
        anatomy = AorticArchAnatomy(
            arch_type="I",
            seed=1000,
            visualization_mesh_path=Path(
                "data/anatomy_registry/anatomies/Tree_00/mesh/visumesh.obj"
            ),
        )

        vessel_tree = build_aortic_arch(anatomy)

        self.assertIsInstance(vessel_tree, AorticArch)
        self.assertEqual(
            vessel_tree.visu_mesh_path,
            "data/anatomy_registry/anatomies/Tree_00/mesh/visumesh.obj",
        )

    def test_build_fluoroscopy_returns_tracking_only_with_mapped_settings(self) -> None:
        anatomy = self._anatomy()
        vessel_tree = build_aortic_arch(anatomy)
        vessel_tree.reset()
        simulation = SimulationDummy()
        spec = FluoroscopySpec(
            image_frequency_hz=12.5,
            image_rot_zx_deg=(15.0, -10.0),
        )

        fluoroscopy = build_fluoroscopy(
            spec=spec,
            vessel_tree=vessel_tree,
            simulation=simulation,
        )

        self.assertIsInstance(fluoroscopy, TrackingOnly)
        self.assertIs(fluoroscopy.vessel_tree, vessel_tree)
        self.assertIs(fluoroscopy.simulation, simulation)
        self.assertEqual(fluoroscopy.image_frequency, 12.5)
        self.assertEqual(fluoroscopy.image_rot_zx, (15.0, -10.0))

    def test_build_target_returns_branch_end(self) -> None:
        vessel_tree, fluoroscopy = self._scene_components()
        spec = BranchEndTarget(
            threshold_mm=4.5,
            branches=("lcca", "rcca"),
        )

        target = build_target(spec, vessel_tree=vessel_tree, fluoroscopy=fluoroscopy)
        target.reset(seed=123)

        self.assertIsInstance(target, BranchEnd)
        self.assertEqual(target.threshold, 4.5)
        self.assertEqual(target.branches, ["lcca", "rcca"])

    def test_build_target_returns_branch_index(self) -> None:
        vessel_tree, fluoroscopy = self._scene_components()
        spec = BranchIndexTarget(
            branch="lcca",
            index=-1,
            threshold_mm=3.0,
        )

        target = build_target(spec, vessel_tree=vessel_tree, fluoroscopy=fluoroscopy)
        target.reset(seed=123)

        self.assertIsInstance(target, BranchIndex)
        self.assertEqual(target.threshold, 3.0)
        self.assertEqual(target.branch, "lcca")
        self.assertEqual(target.idx, -1)

    def test_build_target_returns_manual(self) -> None:
        vessel_tree, fluoroscopy = self._scene_components()
        spec = ManualTarget(
            threshold_mm=2.5,
            targets_vessel_cs=((1.0, 2.0, 3.0),),
        )

        target = build_target(spec, vessel_tree=vessel_tree, fluoroscopy=fluoroscopy)
        target.reset(seed=123)

        self.assertIsInstance(target, Manual)
        self.assertEqual(target.threshold, 2.5)
        self.assertEqual(len(target.targets_vessel_cs), 1)
        np.testing.assert_allclose(
            target.targets_vessel_cs[0],
            np.array([1.0, 2.0, 3.0]),
        )

    def _scene_components(self) -> tuple[AorticArch, TrackingOnly]:
        vessel_tree = build_aortic_arch(self._anatomy())
        vessel_tree.reset()
        fluoroscopy = build_fluoroscopy(
            spec=FluoroscopySpec(),
            vessel_tree=vessel_tree,
            simulation=SimulationDummy(),
        )
        return vessel_tree, fluoroscopy


if __name__ == "__main__":
    unittest.main()
