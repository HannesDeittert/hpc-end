from __future__ import annotations

import unittest

from steve_recommender.eval_v2.models import AnatomyBranch, AorticArchAnatomy
from steve_recommender.eval_v2.target_discovery import AnatomyTargetDiscovery


class AnatomyTargetDiscoveryTests(unittest.TestCase):
    def _anatomy(self) -> AorticArchAnatomy:
        return AorticArchAnatomy(
            arch_type="II",
            seed=42,
            rotation_yzx_deg=(1.0, 2.0, 3.0),
            scaling_xyzd=(1.0, 1.1, 1.2, 1.3),
            omit_axis="z",
        )

    def test_list_branches_returns_real_branch_descriptors_for_selected_anatomy(self) -> None:
        discovery = AnatomyTargetDiscovery()

        branches = discovery.list_branches(self._anatomy())

        self.assertIsInstance(branches, tuple)
        self.assertTrue(all(isinstance(branch, AnatomyBranch) for branch in branches))
        self.assertEqual(tuple(branch.name for branch in branches), tuple(sorted(branch.name for branch in branches)))

        lcca = discovery.get_branch(self._anatomy(), branch_name="lcca")
        self.assertEqual(lcca.name, "lcca")
        self.assertGreater(lcca.length_mm, 0.0)
        self.assertGreater(lcca.point_count, 1)
        self.assertEqual(lcca.terminal_index, lcca.point_count - 1)
        self.assertEqual(len(lcca.centerline_points_vessel_cs), lcca.point_count)
        self.assertEqual(lcca.start_vessel_cs, lcca.centerline_points_vessel_cs[0])
        self.assertEqual(lcca.end_vessel_cs, lcca.centerline_points_vessel_cs[-1])

    def test_get_branch_raises_key_error_for_unknown_branch(self) -> None:
        discovery = AnatomyTargetDiscovery()

        with self.assertRaises(KeyError):
            discovery.get_branch(self._anatomy(), branch_name="missing_branch")

    def test_list_target_modes_exposes_current_supported_target_kinds(self) -> None:
        discovery = AnatomyTargetDiscovery()

        modes = discovery.list_target_modes()
        mode_map = {mode.kind: mode for mode in modes}

        self.assertEqual(tuple(mode_map.keys()), ("branch_end", "branch_index", "manual"))
        self.assertTrue(mode_map["branch_end"].requires_branch_selection)
        self.assertFalse(mode_map["branch_end"].requires_index_selection)
        self.assertTrue(mode_map["branch_end"].allows_multi_branch_selection)
        self.assertFalse(mode_map["branch_end"].requires_manual_points)

        self.assertTrue(mode_map["branch_index"].requires_branch_selection)
        self.assertTrue(mode_map["branch_index"].requires_index_selection)
        self.assertFalse(mode_map["branch_index"].allows_multi_branch_selection)
        self.assertFalse(mode_map["branch_index"].requires_manual_points)

        self.assertFalse(mode_map["manual"].requires_branch_selection)
        self.assertFalse(mode_map["manual"].requires_index_selection)
        self.assertFalse(mode_map["manual"].allows_multi_branch_selection)
        self.assertTrue(mode_map["manual"].requires_manual_points)


if __name__ == "__main__":
    unittest.main()
