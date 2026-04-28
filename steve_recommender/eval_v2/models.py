from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Union


Vector3 = Tuple[float, float, float]
RotationYZXDeg = Tuple[float, float, float]
ScalingXYZD = Tuple[float, float, float, float]
TargetModeKind = Literal["branch_end", "branch_index", "manual"]

PolicySourceKind = Literal["registry", "explicit"]
PolicyMode = Literal["deterministic", "stochastic"]
StochasticEnvironmentMode = Literal["fixed_start", "random_start"]
SimulationBackend = Literal["single_process", "multiprocess"]
ForceTelemetryMode = Literal[
    "passive",
    "intrusive_lcp",
    "constraint_projected_si_validated",
]
ScoringMode = Literal["default_v1"]


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_tuple_length(
    value: Optional[Tuple[float, ...]],
    *,
    expected: int,
    field_name: str,
) -> None:
    if value is not None and len(value) != expected:
        raise ValueError(f"{field_name} must have length {expected}, got {len(value)}")


def _require_positive(value: float, *, field_name: str) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{field_name} must be > 0")


def _require_non_negative(value: float, *, field_name: str) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{field_name} must be >= 0")


def _default_tip_threshold_mm() -> float:
    from .force_telemetry import DEFAULT_TIP_THRESHOLD_MM

    return float(DEFAULT_TIP_THRESHOLD_MM)


@dataclass(frozen=True)
class WireRef:
    """Canonical physical-wire identifier.

    Maps to the current storage layout:
    `data/wire_registry/<model>/wire_versions/<wire>/tool.py`
    """

    model: str
    wire: str

    def __post_init__(self) -> None:
        _require_non_empty(self.model, field_name="model")
        _require_non_empty(self.wire, field_name="wire")

    @property
    def tool_ref(self) -> str:
        return f"{self.model}/{self.wire}"


@dataclass(frozen=True)
class AgentRef:
    """Canonical registered-agent identifier.

    Maps to the current storage layout:
    `data/wire_registry/<model>/wire_versions/<wire>/agents/<agent>/agent.json`
    """

    wire: WireRef
    agent: str

    def __post_init__(self) -> None:
        _require_non_empty(self.agent, field_name="agent")

    @property
    def agent_ref(self) -> str:
        return f"{self.wire.tool_ref}:{self.agent}"


@dataclass(frozen=True)
class PolicySpec:
    """Resolved policy artifact plus provenance.

    `trained_on_wire` is intentionally separate from the execution wire used
    during evaluation so eval_v2 can represent cross-wire experiments directly.
    """

    name: str
    checkpoint_path: Path
    source: PolicySourceKind = "explicit"
    trained_on_wire: Optional[WireRef] = None
    registry_agent: Optional[AgentRef] = None
    metadata_path: Optional[Path] = None
    run_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        _require_non_empty(self.name, field_name="name")
        if self.registry_agent is not None and self.trained_on_wire is None:
            object.__setattr__(self, "trained_on_wire", self.registry_agent.wire)

    @property
    def agent_ref(self) -> Optional[str]:
        if self.registry_agent is None:
            return None
        return self.registry_agent.agent_ref


@dataclass(frozen=True)
class EvaluationCandidate:
    """One policy evaluated on one physical wire."""

    name: str
    execution_wire: WireRef
    policy: PolicySpec

    def __post_init__(self) -> None:
        _require_non_empty(self.name, field_name="name")

    @property
    def is_cross_wire(self) -> bool:
        trained_on = self.policy.trained_on_wire
        return trained_on is not None and trained_on != self.execution_wire


@dataclass(frozen=True)
class AorticArchAnatomy:
    """Resolved anatomy definition for `eve.intervention.vesseltree.AorticArch`.

    This combines the reproducibility fields from `AorticArchRecord` with the
    geometry fields consumed by the current evaluation factory.
    """

    anatomy_type: Literal["aortic_arch"] = "aortic_arch"
    arch_type: str = "I"
    seed: int = 30
    rotation_yzx_deg: Optional[RotationYZXDeg] = None
    scaling_xyzd: Optional[ScalingXYZD] = None
    omit_axis: Optional[str] = None
    record_id: Optional[str] = None
    created_at: str = ""
    centerline_bundle_path: Optional[Path] = None
    simulation_mesh_path: Optional[Path] = None
    visualization_mesh_path: Optional[Path] = None

    def __post_init__(self) -> None:
        _require_non_empty(self.arch_type, field_name="arch_type")
        _require_tuple_length(
            self.rotation_yzx_deg,
            expected=3,
            field_name="rotation_yzx_deg",
        )
        _require_tuple_length(
            self.scaling_xyzd,
            expected=4,
            field_name="scaling_xyzd",
        )


@dataclass(frozen=True)
class AnatomyBranch:
    """Resolved centerline branch available for target selection."""

    name: str
    centerline_points_vessel_cs: Tuple[Vector3, ...]
    length_mm: float

    def __post_init__(self) -> None:
        _require_non_empty(self.name, field_name="name")
        if not self.centerline_points_vessel_cs:
            raise ValueError("centerline_points_vessel_cs must not be empty")
        for point in self.centerline_points_vessel_cs:
            _require_tuple_length(
                point,
                expected=3,
                field_name="centerline_points_vessel_cs",
            )
        _require_positive(self.length_mm, field_name="length_mm")

    @property
    def point_count(self) -> int:
        return len(self.centerline_points_vessel_cs)

    @property
    def terminal_index(self) -> int:
        return self.point_count - 1

    @property
    def start_vessel_cs(self) -> Vector3:
        return self.centerline_points_vessel_cs[0]

    @property
    def end_vessel_cs(self) -> Vector3:
        return self.centerline_points_vessel_cs[-1]


@dataclass(frozen=True)
class TargetModeDescriptor:
    """UI/CLI-facing descriptor for a supported target construction mode."""

    kind: TargetModeKind
    label: str
    description: str
    requires_branch_selection: bool
    requires_index_selection: bool
    allows_multi_branch_selection: bool
    requires_manual_points: bool

    def __post_init__(self) -> None:
        _require_non_empty(self.label, field_name="label")
        _require_non_empty(self.description, field_name="description")


@dataclass(frozen=True)
class FluoroscopySpec:
    """Parameters passed to the tracking-based fluoroscopy backend."""

    image_frequency_hz: float = 7.5
    image_rot_zx_deg: Tuple[float, float] = (20.0, 5.0)

    def __post_init__(self) -> None:
        _require_positive(self.image_frequency_hz, field_name="image_frequency_hz")
        _require_tuple_length(
            self.image_rot_zx_deg,
            expected=2,
            field_name="image_rot_zx_deg",
        )


@dataclass(frozen=True)
class BranchEndTarget:
    """Branch-end target for `eve.intervention.target.BranchEnd`.

    If multiple branches are supplied, upstream stEVE chooses one endpoint from
    the provided set.
    """

    kind: Literal["branch_end"] = "branch_end"
    threshold_mm: float = 5.0
    branches: Tuple[str, ...] = ("lcca",)

    def __post_init__(self) -> None:
        _require_positive(self.threshold_mm, field_name="threshold_mm")
        if not self.branches:
            raise ValueError("branches must not be empty")
        for branch in self.branches:
            _require_non_empty(branch, field_name="branches")


@dataclass(frozen=True)
class BranchIndexTarget:
    """Exact centerline-index target for `eve.intervention.target.BranchIndex`."""

    kind: Literal["branch_index"] = "branch_index"
    branch: str = "lcca"
    index: int = -1
    threshold_mm: float = 5.0

    def __post_init__(self) -> None:
        _require_non_empty(self.branch, field_name="branch")
        _require_positive(self.threshold_mm, field_name="threshold_mm")


@dataclass(frozen=True)
class ManualTarget:
    """Manual vessel-coordinate targets for `eve.intervention.target.Manual`."""

    kind: Literal["manual"] = "manual"
    targets_vessel_cs: Tuple[Vector3, ...] = ()
    threshold_mm: float = 5.0

    def __post_init__(self) -> None:
        _require_positive(self.threshold_mm, field_name="threshold_mm")
        if not self.targets_vessel_cs:
            raise ValueError("targets_vessel_cs must not be empty")
        for target in self.targets_vessel_cs:
            _require_tuple_length(
                target,
                expected=3,
                field_name="targets_vessel_cs",
            )


TargetSpec = Union[BranchEndTarget, BranchIndexTarget, ManualTarget]


@dataclass(frozen=True)
class ForceUnits:
    """Explicit physical-unit metadata for validated SI conversion."""

    length_unit: Literal["mm", "m"] = "mm"
    mass_unit: Literal["kg", "g"] = "kg"
    time_unit: Literal["s", "ms"] = "s"


@dataclass(frozen=True)
class ForceCalibrationPolicy:
    """Calibration-cache policy for validated force telemetry."""

    required: bool = True
    cache_path: Path = Path("results/force_calibration/cache.json")
    tolerance_profile: str = "default_v1"

    def __post_init__(self) -> None:
        _require_non_empty(self.tolerance_profile, field_name="tolerance_profile")


@dataclass(frozen=True)
class ForceTelemetrySpec:
    """Wall-force extraction settings for `SofaWallForceInfo`.

    `tip_threshold_mm` is an absolute arc length from the distal end of the
    wire, in mm. A collision DOF is classified as tip when its arc-length
    distance from the distal end is less than or equal to this positive
    threshold.
    """

    mode: ForceTelemetryMode = "passive"
    required: bool = False
    contact_epsilon: float = 1e-7
    tip_threshold_mm: float = field(default_factory=_default_tip_threshold_mm)
    plugin_path: Optional[Path] = None
    units: Optional[ForceUnits] = None
    calibration: ForceCalibrationPolicy = field(
        default_factory=ForceCalibrationPolicy
    )

    def __post_init__(self) -> None:
        _require_non_negative(
            self.contact_epsilon,
            field_name="contact_epsilon",
        )
        if float(self.tip_threshold_mm) <= 0.0:
            raise ValueError(
                f"tip_threshold_mm must be > 0, got {float(self.tip_threshold_mm)}"
            )
        if self.mode == "constraint_projected_si_validated" and self.units is None:
            raise ValueError(
                "units are required when mode='constraint_projected_si_validated'"
            )


@dataclass(frozen=True)
class VisualizationSpec:
    """Optional evaluation-time rendering settings."""

    enabled: bool = False
    rendered_trials_per_candidate: int = 1
    force_debug_overlay: bool = False
    force_debug_top_k_segments: int = 5

    def __post_init__(self) -> None:
        if self.rendered_trials_per_candidate < 1:
            raise ValueError("rendered_trials_per_candidate must be >= 1")
        if self.force_debug_top_k_segments < 1:
            raise ValueError("force_debug_top_k_segments must be >= 1")


@dataclass(frozen=True)
class ExecutionPlan:
    """How scenarios are executed once candidates are resolved.

    Seed management intentionally distinguishes between two independent sources
    of randomness:

    - environment seeds control reproducible reset state such as initial wire
      orientation or insertion randomness inside the intervention scene,
    - policy seeds control stochastic action sampling when `policy_mode` is
      `"stochastic"`.

    Every candidate in one job shares the exact same derived seed schedules so
    candidate comparisons remain paired trial-by-trial.
    """

    trials_per_candidate: int = 10
    base_seed: int = 123
    explicit_seeds: Tuple[int, ...] = ()
    policy_base_seed: int = 1000
    policy_explicit_seeds: Tuple[int, ...] = ()
    max_episode_steps: int = 1000
    policy_device: str = "cuda"
    policy_mode: PolicyMode = "deterministic"
    stochastic_environment_mode: StochasticEnvironmentMode = "random_start"
    simulation_backend: SimulationBackend = "single_process"
    visualization: VisualizationSpec = field(default_factory=VisualizationSpec)
    worker_count: int = 1

    def __post_init__(self) -> None:
        if self.trials_per_candidate < 1:
            raise ValueError("trials_per_candidate must be >= 1")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1")
        if self.worker_count < 1:
            raise ValueError("worker_count must be >= 1")
        if self.visualization.enabled and self.worker_count > 1:
            raise ValueError("worker_count > 1 is only supported for headless runs")
        _require_non_empty(self.policy_device, field_name="policy_device")
        if self.explicit_seeds and len(self.explicit_seeds) != self.trials_per_candidate:
            raise ValueError(
                "explicit_seeds must match trials_per_candidate "
                f"({self.trials_per_candidate}), got {len(self.explicit_seeds)}"
            )
        if self.policy_explicit_seeds and len(self.policy_explicit_seeds) != self.trials_per_candidate:
            raise ValueError(
                "policy_explicit_seeds must match trials_per_candidate "
                f"({self.trials_per_candidate}), got {len(self.policy_explicit_seeds)}"
            )

    @property
    def seeds(self) -> Tuple[int, ...]:
        """Backward-compatible alias for environment seeds."""

        return self.environment_seeds

    @property
    def environment_seeds(self) -> Tuple[int, ...]:
        """Return the per-trial environment seed schedule.

        Rules:
        - `explicit_seeds` wins when provided.
        - deterministic mode uses an incrementing sequence from `base_seed`.
        - stochastic `fixed_start` keeps the environment constant across trials.
        - stochastic `random_start` uses the incrementing sequence.
        """

        if self.explicit_seeds:
            return self.explicit_seeds
        if self.policy_mode == "stochastic" and self.stochastic_environment_mode == "fixed_start":
            return tuple(self.base_seed for _ in range(self.trials_per_candidate))
        return tuple(self.base_seed + offset for offset in range(self.trials_per_candidate))

    @property
    def policy_seeds(self) -> Tuple[Optional[int], ...]:
        """Return the per-trial policy seed schedule.

        Deterministic policies disable policy sampling, so they do not carry
        independent policy seeds. Stochastic policies either use the explicit
        override or an incrementing sequence from `policy_base_seed`.
        """

        if self.policy_mode == "deterministic":
            return tuple(None for _ in range(self.trials_per_candidate))
        if self.policy_explicit_seeds:
            return tuple(int(seed) for seed in self.policy_explicit_seeds)
        return tuple(self.policy_base_seed + offset for offset in range(self.trials_per_candidate))

    @property
    def trial_seed_pairs(self) -> Tuple[Tuple[int, Optional[int]], ...]:
        """Pair environment and policy seed schedules trial-by-trial."""

        return tuple(zip(self.environment_seeds, self.policy_seeds))

    def environment_seed_for_trial(self, trial_index: int) -> int:
        """Resolve one environment seed for an arbitrary trial index.

        The service normally iterates the precomputed schedule, but tests and
        lower-level helpers may call into the runner directly with any
        `trial_index`. This helper keeps that path deterministic and consistent
        with the configured schedule semantics.
        """

        normalized_index = max(int(trial_index), 0)
        if self.explicit_seeds:
            if normalized_index < len(self.explicit_seeds):
                return int(self.explicit_seeds[normalized_index])
            return int(self.explicit_seeds[-1])
        if self.policy_mode == "stochastic" and self.stochastic_environment_mode == "fixed_start":
            return int(self.base_seed)
        return int(self.base_seed + normalized_index)

    def policy_seed_for_trial(self, trial_index: int) -> Optional[int]:
        """Resolve one policy seed for an arbitrary trial index."""

        if self.policy_mode == "deterministic":
            return None
        normalized_index = max(int(trial_index), 0)
        if self.policy_explicit_seeds:
            if normalized_index < len(self.policy_explicit_seeds):
                return int(self.policy_explicit_seeds[normalized_index])
            return int(self.policy_explicit_seeds[-1])
        return int(self.policy_base_seed + normalized_index)


@dataclass(frozen=True)
class ScoreWeights:
    """Category weights used by the current `default_v1` score."""

    success: float = 2.0
    efficiency: float = 1.0
    safety: float = 1.0
    smoothness: float = 0.25
    normalize: bool = True


@dataclass(frozen=True)
class ScoreScales:
    """Scale parameters used by the current `default_v1` score."""

    force_scale: float = 1.0
    lcp_scale: float = 1.0
    speed_scale_mm_s: float = 50.0

    def __post_init__(self) -> None:
        _require_positive(self.force_scale, field_name="force_scale")
        _require_positive(self.lcp_scale, field_name="lcp_scale")
        _require_positive(self.speed_scale_mm_s, field_name="speed_scale_mm_s")


@dataclass(frozen=True)
class ScoringSpec:
    """Current scoring configuration, grouped for GUI slider control."""

    mode: ScoringMode = "default_v1"
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    scales: ScoreScales = field(default_factory=ScoreScales)


@dataclass(frozen=True)
class EvaluationScenario:
    """One shared intervention scene evaluated by one or more candidates.

    Maps directly to the objects built by the current evaluation pipeline:
    - `anatomy` -> `eve.intervention.vesseltree.AorticArch`
    - `fluoroscopy` -> tracking-based fluoroscopy (TrackingOnly-compatible)
    - `target` -> `eve.intervention.target.*`
    - `friction` -> `SofaBeamAdapter(friction=...)`
    - `stop_device_at_tree_end` / `normalize_action` -> `MonoPlaneStatic(...)`
    """

    name: str
    anatomy: AorticArchAnatomy
    target: TargetSpec
    fluoroscopy: FluoroscopySpec = field(default_factory=FluoroscopySpec)
    friction: float = 0.001
    stop_device_at_tree_end: bool = True
    normalize_action: bool = True
    force_telemetry: ForceTelemetrySpec = field(default_factory=ForceTelemetrySpec)

    def __post_init__(self) -> None:
        _require_non_empty(self.name, field_name="name")
        _require_non_negative(self.friction, field_name="friction")

    @property
    def action_dt_s(self) -> float:
        return 1.0 / float(self.fluoroscopy.image_frequency_hz)


@dataclass(frozen=True)
class EvaluationJob:
    """Top-level eval_v2 request passed into the service layer."""

    name: str
    scenarios: Tuple[EvaluationScenario, ...]
    candidates: Tuple[EvaluationCandidate, ...]
    execution: ExecutionPlan = field(default_factory=ExecutionPlan)
    scoring: ScoringSpec = field(default_factory=ScoringSpec)
    output_root: Path = Path("results/eval_runs")

    def __post_init__(self) -> None:
        _require_non_empty(self.name, field_name="name")
        if not self.scenarios:
            raise ValueError("scenarios must not be empty")
        if not self.candidates:
            raise ValueError("candidates must not be empty")


@dataclass(frozen=True)
class ScoreBreakdown:
    """Per-trial score and its current component scores."""

    total: float
    success: float
    efficiency: float
    safety: Optional[float]
    smoothness: float


@dataclass(frozen=True)
class ForceTelemetrySummary:
    """Per-trial force summary derived from `SofaWallForceInfo` + pipeline reducers."""

    available_for_score: bool
    validation_status: str
    validation_error: Optional[str] = None
    source: str = ""
    channel: str = ""
    quality_tier: str = "unavailable"
    association_method: str = "none"
    association_explicit_ratio: Optional[float] = None
    association_coverage: Optional[float] = None
    association_explicit_force_coverage: Optional[float] = None
    ordering_stable: bool = False
    active_constraint_any: bool = False
    contact_detected_any: bool = False
    contact_count_max: int = 0
    segment_count_max: int = 0
    lcp_max_abs_max: Optional[float] = None
    lcp_sum_abs_mean: Optional[float] = None
    wire_force_norm_max: Optional[float] = None
    wire_force_norm_mean: Optional[float] = None
    collision_force_norm_max: Optional[float] = None
    collision_force_norm_mean: Optional[float] = None
    total_force_norm_max: Optional[float] = None
    total_force_norm_mean: Optional[float] = None
    total_force_norm_max_newton: Optional[float] = None
    total_force_norm_mean_newton: Optional[float] = None
    peak_segment_force_norm: Optional[float] = None
    peak_segment_force_norm_newton: Optional[float] = None
    peak_segment_force_step: Optional[int] = None
    peak_segment_force_segment_id: Optional[int] = None
    peak_segment_force_time_s: Optional[float] = None
    gap_active_projected_count_sum: int = 0
    gap_explicit_mapped_count_sum: int = 0
    gap_unmapped_count_sum: int = 0
    gap_unmapped_ratio: Optional[float] = None
    gap_dominant_class: str = "none"
    gap_contact_mode: str = "none"
    tip_force_available: bool = False
    tip_force_validation_status: str = "unmapped"
    tip_force_records: Tuple[dict, ...] = ()
    tip_force_total_vector_N: Vector3 = (0.0, 0.0, 0.0)
    tip_force_total_norm_N: float = 0.0
    lcp_mapped_wall_row_count_max: int = 0
    lcp_contact_export_coverage: Optional[float] = None


@dataclass(frozen=True)
class TrialTelemetrySummary:
    """Per-trial summary metrics emitted by the current pipeline."""

    success: bool
    steps_total: int
    steps_to_success: Optional[int]
    episode_reward: float
    wall_time_s: Optional[float] = None
    sim_time_s: Optional[float] = None
    path_ratio_last: Optional[float] = None
    trajectory_length_last: Optional[float] = None
    average_translation_speed_last: Optional[float] = None
    tip_speed_max_mm_s: Optional[float] = None
    tip_speed_mean_mm_s: Optional[float] = None
    forces: Optional[ForceTelemetrySummary] = None


@dataclass(frozen=True)
class TrialArtifactPaths:
    """Filesystem artifacts emitted for one executed trial."""

    trace_npz_path: Optional[Path] = None
    force_gap_json_path: Optional[Path] = None
    force_gap_csv_path: Optional[Path] = None


@dataclass(frozen=True)
class TrialResult:
    """One candidate/scenario/trial-seed execution result."""

    scenario_name: str
    candidate_name: str
    execution_wire: WireRef
    policy: PolicySpec
    trial_index: int
    seed: int
    score: ScoreBreakdown
    telemetry: TrialTelemetrySummary
    policy_seed: Optional[int] = None
    artifacts: TrialArtifactPaths = field(default_factory=TrialArtifactPaths)


@dataclass(frozen=True)
class CandidateSummary:
    """Aggregate summary for one candidate on one scenario."""

    scenario_name: str
    candidate_name: str
    execution_wire: WireRef
    trained_on_wire: Optional[WireRef]
    trial_count: int
    success_rate: Optional[float]
    score_mean: Optional[float]
    score_std: Optional[float]
    steps_total_mean: Optional[float]
    steps_to_success_mean: Optional[float]
    tip_speed_max_mean_mm_s: Optional[float]
    wall_force_max_mean: Optional[float]
    wall_force_max_mean_newton: Optional[float]
    force_available_rate: Optional[float]


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Top-level files written by an evaluation run."""

    output_dir: Path
    summary_csv_path: Optional[Path] = None
    report_json_path: Optional[Path] = None
    report_markdown_path: Optional[Path] = None


@dataclass(frozen=True)
class EvaluationReport:
    """Service-layer output returned after a completed evaluation job."""

    job_name: str
    generated_at: str
    summaries: Tuple[CandidateSummary, ...]
    trials: Tuple[TrialResult, ...]
    artifacts: Optional[EvaluationArtifacts] = None


@dataclass(frozen=True)
class HistoricalReportSummary:
    """Lightweight metadata for one persisted report on disk."""

    job_name: str
    generated_at: str
    anatomy: str
    tested_wires: Tuple[str, ...]
    report_json_path: Path
    output_dir: Path


__all__ = [
    "AnatomyBranch",
    "AorticArchAnatomy",
    "AgentRef",
    "BranchEndTarget",
    "BranchIndexTarget",
    "CandidateSummary",
    "EvaluationArtifacts",
    "EvaluationCandidate",
    "EvaluationJob",
    "EvaluationReport",
    "EvaluationScenario",
    "ExecutionPlan",
    "FluoroscopySpec",
    "ForceCalibrationPolicy",
    "ForceTelemetryMode",
    "ForceTelemetrySpec",
    "ForceTelemetrySummary",
    "HistoricalReportSummary",
    "ForceUnits",
    "ManualTarget",
    "PolicySpec",
    "ScoreBreakdown",
    "ScoreScales",
    "ScoreWeights",
    "ScoringSpec",
    "TargetSpec",
    "TargetModeDescriptor",
    "TrialArtifactPaths",
    "TrialResult",
    "TrialTelemetrySummary",
    "WireRef",
]
