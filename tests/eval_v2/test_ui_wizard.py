from __future__ import annotations

import unittest

from steve_recommender.eval_v2.models import AorticArchAnatomy, WireRef
from steve_recommender.eval_v2.ui_controller import ClinicalUIController, WizardState


class _WizardServiceStub:
    def list_anatomies(self, *, registry_path=None):
        _ = registry_path
        return ()

    def list_branches(self, anatomy):
        _ = anatomy
        return ()

    def list_target_modes(self):
        return ()

    def list_execution_wires(self):
        return ()

    def list_startable_wires(self):
        return ()

    def list_registry_policies(self, *, execution_wire=None):
        _ = execution_wire
        return ()

    def list_explicit_policies(self, *, execution_wire=None):
        _ = execution_wire
        return ()

    def list_candidates(self, *, execution_wire, include_cross_wire=True):
        _ = execution_wire, include_cross_wire
        return ()

    def run_evaluation_job(self, job, *, frame_callback=None, progress_callback=None):
        _ = job, frame_callback, progress_callback
        raise RuntimeError("not used")


class WizardStateTests(unittest.TestCase):
    def test_default_state_values(self) -> None:
        state = WizardState()

        self.assertIsNone(state.anatomy)
        self.assertEqual(state.branch, "")
        self.assertIsNone(state.target_position)
        self.assertEqual(state.selected_wires, [])
        self.assertTrue(state.is_deterministic)
        self.assertEqual(state.trials_per_wire, 1)
        self.assertFalse(state.is_visualized)
        self.assertEqual(state.visualized_trials_count, 1)


class ClinicalUIControllerWizardStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = ClinicalUIController(service=_WizardServiceStub())

    def test_reset_wizard_state_clears_all_fields(self) -> None:
        anatomy = AorticArchAnatomy(arch_type="II", seed=42, record_id="Tree_00")
        wire = WireRef(model="steve_default", wire="standard_j")

        self.controller.set_wizard_anatomy(anatomy)
        self.controller.set_wizard_branch("lcca")
        self.controller.set_wizard_target_position(0.75)
        self.controller.set_wizard_selected_wires((wire,))
        self.controller.set_wizard_execution_config(
            is_deterministic=False,
            trials_per_wire=3,
            is_visualized=True,
            visualized_trials_count=2,
        )

        self.controller.reset_wizard_state()
        state = self.controller.get_wizard_state()

        self.assertIsNone(state.anatomy)
        self.assertEqual(state.branch, "")
        self.assertIsNone(state.target_position)
        self.assertEqual(state.selected_wires, [])
        self.assertTrue(state.is_deterministic)
        self.assertEqual(state.trials_per_wire, 1)
        self.assertFalse(state.is_visualized)
        self.assertEqual(state.visualized_trials_count, 1)

    def test_validation_blocks_forward_without_selected_wires(self) -> None:
        anatomy = AorticArchAnatomy(arch_type="II", seed=42, record_id="Tree_00")

        self.controller.set_wizard_anatomy(anatomy)
        self.controller.set_wizard_branch("lcca")
        self.controller.set_wizard_target_position(1.0)
        self.controller.set_wizard_selected_wires(())

        self.assertFalse(self.controller.can_forward_from_step(3))

        wire = WireRef(model="steve_default", wire="standard_j")
        self.controller.set_wizard_selected_wires((wire,))
        self.assertTrue(self.controller.can_forward_from_step(3))

    def test_execution_validation_checks_visualized_trial_count_bounds(self) -> None:
        self.controller.set_wizard_execution_config(
            is_deterministic=False,
            trials_per_wire=2,
            is_visualized=True,
            visualized_trials_count=3,
        )
        self.assertFalse(self.controller.can_forward_from_step(4))

        self.controller.set_wizard_execution_config(
            is_deterministic=False,
            trials_per_wire=2,
            is_visualized=True,
            visualized_trials_count=2,
        )
        self.assertTrue(self.controller.can_forward_from_step(4))


if __name__ == "__main__":
    unittest.main()