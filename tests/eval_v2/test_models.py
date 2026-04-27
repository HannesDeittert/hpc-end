from __future__ import annotations

import unittest
from pathlib import Path

from steve_recommender.eval_v2.models import (
    AgentRef,
    AnatomyBranch,
    AorticArchAnatomy,
    BranchEndTarget,
    BranchIndexTarget,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationScenario,
    ExecutionPlan,
    FluoroscopySpec,
    ForceCalibrationPolicy,
    ForceTelemetrySpec,
    ForceUnits,
    ManualTarget,
    PolicySpec,
    ScoreScales,
    TargetModeDescriptor,
    VisualizationSpec,
    WireRef,
)


def _wire(*, model: str = "steve_default", wire: str = "standard_j") -> WireRef:
    return WireRef(model=model, wire=wire)


def _agent_ref(*, wire: WireRef | None = None, agent: str = "archvar_best") -> AgentRef:
    return AgentRef(wire=wire or _wire(), agent=agent)


def _policy(
    *,
    name: str = "policy_a",
    trained_on_wire: WireRef | None = None,
    registry_agent: AgentRef | None = None,
) -> PolicySpec:
    return PolicySpec(
        name=name,
        checkpoint_path=Path("/tmp/checkpoint.everl"),
        trained_on_wire=trained_on_wire,
        registry_agent=registry_agent,
    )


def _candidate(
    *,
    name: str = "candidate_a",
    execution_wire: WireRef | None = None,
    policy: PolicySpec | None = None,
) -> EvaluationCandidate:
    return EvaluationCandidate(
        name=name,
        execution_wire=execution_wire or _wire(),
        policy=policy or _policy(),
    )


def _anatomy(
    *,
    arch_type: str = "I",
    rotation_yzx_deg: tuple[float, ...] | None = None,
    scaling_xyzd: tuple[float, ...] | None = None,
) -> AorticArchAnatomy:
    return AorticArchAnatomy(
        arch_type=arch_type,
        rotation_yzx_deg=rotation_yzx_deg,  # type: ignore[arg-type]
        scaling_xyzd=scaling_xyzd,  # type: ignore[arg-type]
    )


def _scenario(
    *,
    name: str = "scenario_a",
    friction: float = 0.001,
    fluoroscopy: FluoroscopySpec | None = None,
    target: BranchEndTarget | BranchIndexTarget | ManualTarget | None = None,
) -> EvaluationScenario:
    return EvaluationScenario(
        name=name,
        anatomy=_anatomy(),
        target=target or BranchEndTarget(),
        fluoroscopy=fluoroscopy or FluoroscopySpec(),
        friction=friction,
    )


def _job(
    *,
    name: str = "job_a",
    scenarios: tuple[EvaluationScenario, ...] | None = None,
    candidates: tuple[EvaluationCandidate, ...] | None = None,
) -> EvaluationJob:
    return EvaluationJob(
        name=name,
        scenarios=(_scenario(),) if scenarios is None else scenarios,
        candidates=(_candidate(),) if candidates is None else candidates,
    )


class NonEmptyValidationTests(unittest.TestCase):
    def test_non_empty_validations_raise_for_blank_strings(self) -> None:
        cases = [
            ("WireRef.model", lambda: WireRef(model="", wire="standard_j")),
            ("WireRef.wire", lambda: WireRef(model="steve_default", wire="")),
            ("AgentRef.agent", lambda: AgentRef(wire=_wire(), agent="")),
            ("PolicySpec.name", lambda: _policy(name="")),
            ("EvaluationCandidate.name", lambda: _candidate(name="")),
            ("AorticArchAnatomy.arch_type", lambda: _anatomy(arch_type="")),
            ("BranchEndTarget.branches", lambda: BranchEndTarget(branches=("lcca", ""))),
            ("BranchIndexTarget.branch", lambda: BranchIndexTarget(branch="")),
            (
                "ForceCalibrationPolicy.tolerance_profile",
                lambda: ForceCalibrationPolicy(tolerance_profile=""),
            ),
            ("ExecutionPlan.policy_device", lambda: ExecutionPlan(policy_device="")),
            ("EvaluationScenario.name", lambda: _scenario(name="")),
            ("EvaluationJob.name", lambda: _job(name="")),
        ]

        for label, factory in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    factory()


class PositiveAndNonNegativeValidationTests(unittest.TestCase):
    def test_positive_validations_reject_zero_and_negative_values(self) -> None:
        cases = [
            ("FluoroscopySpec.image_frequency_hz.zero", lambda: FluoroscopySpec(image_frequency_hz=0.0)),
            ("FluoroscopySpec.image_frequency_hz.negative", lambda: FluoroscopySpec(image_frequency_hz=-1.0)),
            ("BranchEndTarget.threshold_mm.zero", lambda: BranchEndTarget(threshold_mm=0.0)),
            ("BranchIndexTarget.threshold_mm.negative", lambda: BranchIndexTarget(threshold_mm=-1.0)),
            (
                "ManualTarget.threshold_mm.zero",
                lambda: ManualTarget(targets_vessel_cs=((1.0, 2.0, 3.0),), threshold_mm=0.0),
            ),
            ("ScoreScales.force_scale.zero", lambda: ScoreScales(force_scale=0.0)),
            ("ScoreScales.lcp_scale.negative", lambda: ScoreScales(lcp_scale=-1.0)),
            ("ScoreScales.speed_scale_mm_s.zero", lambda: ScoreScales(speed_scale_mm_s=0.0)),
        ]

        for label, factory in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    factory()

    def test_non_negative_validations_reject_negative_values(self) -> None:
        cases = [
            (
                "ForceTelemetrySpec.contact_epsilon",
                lambda: ForceTelemetrySpec(contact_epsilon=-1e-7),
            ),
            ("EvaluationScenario.friction", lambda: _scenario(friction=-0.001)),
        ]

        for label, factory in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    factory()


class TupleValidationTests(unittest.TestCase):
    def test_tuple_lengths_are_strictly_enforced(self) -> None:
        cases = [
            (
                "AorticArchAnatomy.rotation_yzx_deg",
                lambda: _anatomy(rotation_yzx_deg=(1.0, 2.0)),
            ),
            (
                "AorticArchAnatomy.scaling_xyzd",
                lambda: _anatomy(scaling_xyzd=(1.0, 2.0, 3.0)),
            ),
            (
                "FluoroscopySpec.image_rot_zx_deg",
                lambda: FluoroscopySpec(image_rot_zx_deg=(20.0,)),
            ),
            (
                "ManualTarget.targets_vessel_cs",
                lambda: ManualTarget(targets_vessel_cs=((1.0, 2.0),)),
            ),
            (
                "AnatomyBranch.centerline_points_vessel_cs",
                lambda: AnatomyBranch(
                    name="lcca",
                    centerline_points_vessel_cs=((1.0, 2.0),),
                    length_mm=10.0,
                ),
            ),
        ]

        for label, factory in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    factory()


class PropertyTests(unittest.TestCase):
    def test_wire_ref_tool_ref_property(self) -> None:
        wire = _wire(model="universal_ii", wire="gentle")
        self.assertEqual(wire.tool_ref, "universal_ii/gentle")

    def test_agent_ref_property(self) -> None:
        agent = _agent_ref(wire=_wire(model="amplatz_super_stiff", wire="standard_j"), agent="best")
        self.assertEqual(agent.agent_ref, "amplatz_super_stiff/standard_j:best")

    def test_policy_agent_ref_and_trained_wire_are_derived_from_registry_agent(self) -> None:
        agent = _agent_ref(wire=_wire(model="steve_default", wire="tight_j"), agent="policy_x")
        policy = _policy(registry_agent=agent)
        self.assertEqual(policy.agent_ref, "steve_default/tight_j:policy_x")
        self.assertEqual(policy.trained_on_wire, agent.wire)

    def test_evaluation_candidate_is_cross_wire_true_when_execution_differs(self) -> None:
        trained_on = _wire(model="steve_default", wire="standard_j")
        execution = _wire(model="universal_ii", wire="standard_j")
        candidate = _candidate(
            execution_wire=execution,
            policy=_policy(trained_on_wire=trained_on),
        )
        self.assertTrue(candidate.is_cross_wire)

    def test_evaluation_candidate_is_cross_wire_false_when_execution_matches(self) -> None:
        trained_on = _wire(model="steve_default", wire="standard_j")
        candidate = _candidate(
            execution_wire=trained_on,
            policy=_policy(trained_on_wire=trained_on),
        )
        self.assertFalse(candidate.is_cross_wire)

    def test_evaluation_candidate_is_cross_wire_false_when_policy_wire_unknown(self) -> None:
        candidate = _candidate(policy=_policy(trained_on_wire=None))
        self.assertFalse(candidate.is_cross_wire)

    def test_execution_plan_seeds_property_uses_explicit_seeds_when_provided(self) -> None:
        plan = ExecutionPlan(trials_per_candidate=3, explicit_seeds=(11, 13, 17))
        self.assertEqual(plan.seeds, (11, 13, 17))

    def test_execution_plan_seeds_property_generates_default_sequence(self) -> None:
        plan = ExecutionPlan(trials_per_candidate=3, base_seed=100)
        self.assertEqual(plan.seeds, (100, 101, 102))

    def test_execution_plan_environment_seeds_use_explicit_override(self) -> None:
        plan = ExecutionPlan(
            trials_per_candidate=3,
            explicit_seeds=(11, 13, 17),
        )
        self.assertEqual(plan.environment_seeds, (11, 13, 17))

    def test_execution_plan_rejects_explicit_environment_seed_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit_seeds"):
            ExecutionPlan(
                trials_per_candidate=3,
                explicit_seeds=(11, 13),
            )

    def test_execution_plan_deterministic_policy_seeds_are_disabled(self) -> None:
        plan = ExecutionPlan(
            trials_per_candidate=3,
            base_seed=100,
            policy_mode="deterministic",
        )
        self.assertEqual(plan.policy_seeds, (None, None, None))

    def test_execution_plan_stochastic_policy_seeds_generate_default_sequence(self) -> None:
        plan = ExecutionPlan(
            trials_per_candidate=3,
            base_seed=100,
            policy_mode="stochastic",
            policy_base_seed=1000,
        )
        self.assertEqual(plan.environment_seeds, (100, 101, 102))
        self.assertEqual(plan.policy_seeds, (1000, 1001, 1002))

    def test_execution_plan_stochastic_fixed_start_repeats_environment_seed(self) -> None:
        plan = ExecutionPlan(
            trials_per_candidate=3,
            base_seed=123,
            policy_mode="stochastic",
            stochastic_environment_mode="fixed_start",
        )
        self.assertEqual(plan.environment_seeds, (123, 123, 123))

    def test_execution_plan_stochastic_environment_mode_does_not_override_explicit_seeds(self) -> None:
        plan = ExecutionPlan(
            trials_per_candidate=3,
            explicit_seeds=(123, 999, 42),
            policy_mode="stochastic",
            stochastic_environment_mode="fixed_start",
        )
        self.assertEqual(plan.environment_seeds, (123, 999, 42))

    def test_execution_plan_accepts_explicit_policy_seed_override(self) -> None:
        plan = ExecutionPlan(
            trials_per_candidate=3,
            policy_mode="stochastic",
            policy_explicit_seeds=(1000, 1005, 1010),
        )
        self.assertEqual(plan.policy_seeds, (1000, 1005, 1010))

    def test_execution_plan_rejects_explicit_policy_seed_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "policy_explicit_seeds"):
            ExecutionPlan(
                trials_per_candidate=3,
                policy_mode="stochastic",
                policy_explicit_seeds=(1000, 1001),
            )

    def test_execution_plan_trial_seed_pairs_align_environment_and_policy_sequences(self) -> None:
        plan = ExecutionPlan(
            trials_per_candidate=3,
            base_seed=123,
            policy_mode="stochastic",
            policy_base_seed=1000,
            stochastic_environment_mode="fixed_start",
        )
        self.assertEqual(
            plan.trial_seed_pairs,
            (
                (123, 1000),
                (123, 1001),
                (123, 1002),
            ),
        )

    def test_execution_plan_accepts_default_worker_count(self) -> None:
        plan = ExecutionPlan()
        self.assertEqual(plan.worker_count, 1)

    def test_execution_plan_rejects_invalid_worker_count(self) -> None:
        with self.assertRaises(ValueError):
            ExecutionPlan(worker_count=0)

    def test_execution_plan_rejects_parallel_visualized_runs(self) -> None:
        with self.assertRaises(ValueError):
            ExecutionPlan(
                worker_count=2,
                visualization=VisualizationSpec(enabled=True),
            )

    def test_evaluation_scenario_action_dt_s_property(self) -> None:
        scenario = _scenario(fluoroscopy=FluoroscopySpec(image_frequency_hz=5.0))
        self.assertAlmostEqual(scenario.action_dt_s, 0.2)

    def test_anatomy_branch_properties(self) -> None:
        branch = AnatomyBranch(
            name="lcca",
            centerline_points_vessel_cs=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
            length_mm=12.5,
        )
        self.assertEqual(branch.point_count, 2)
        self.assertEqual(branch.terminal_index, 1)
        self.assertEqual(branch.start_vessel_cs, (1.0, 2.0, 3.0))
        self.assertEqual(branch.end_vessel_cs, (4.0, 5.0, 6.0))


class ForceTelemetrySpecValidationTests(unittest.TestCase):
    def test_validated_force_mode_requires_units(self) -> None:
        with self.assertRaises(ValueError):
            ForceTelemetrySpec(mode="constraint_projected_si_validated")

    def test_validated_force_mode_accepts_units(self) -> None:
        spec = ForceTelemetrySpec(
            mode="constraint_projected_si_validated",
            units=ForceUnits(length_unit="mm", mass_unit="kg", time_unit="s"),
        )
        self.assertEqual(spec.mode, "constraint_projected_si_validated")
        self.assertIsNotNone(spec.units)


class CollectionValidationTests(unittest.TestCase):
    def test_evaluation_job_requires_non_empty_scenarios(self) -> None:
        with self.assertRaises(ValueError):
            _job(scenarios=())

    def test_evaluation_job_requires_non_empty_candidates(self) -> None:
        with self.assertRaises(ValueError):
            _job(candidates=())

    def test_anatomy_branch_requires_non_empty_name_and_points(self) -> None:
        with self.assertRaises(ValueError):
            AnatomyBranch(
                name="",
                centerline_points_vessel_cs=((1.0, 2.0, 3.0),),
                length_mm=10.0,
            )

        with self.assertRaises(ValueError):
            AnatomyBranch(
                name="lcca",
                centerline_points_vessel_cs=(),
                length_mm=10.0,
            )

    def test_target_mode_descriptor_requires_label_and_description(self) -> None:
        with self.assertRaises(ValueError):
            TargetModeDescriptor(
                kind="branch_end",
                label="",
                description="Select branch endpoints",
                requires_branch_selection=True,
                requires_index_selection=False,
                allows_multi_branch_selection=True,
                requires_manual_points=False,
            )

        with self.assertRaises(ValueError):
            TargetModeDescriptor(
                kind="manual",
                label="Manual Coordinates",
                description="",
                requires_branch_selection=False,
                requires_index_selection=False,
                allows_multi_branch_selection=False,
                requires_manual_points=True,
            )

    def test_branch_end_target_requires_non_empty_branches(self) -> None:
        with self.assertRaises(ValueError):
            BranchEndTarget(branches=())

    def test_manual_target_requires_non_empty_targets(self) -> None:
        with self.assertRaises(ValueError):
            ManualTarget(targets_vessel_cs=())


if __name__ == "__main__":
    unittest.main()
