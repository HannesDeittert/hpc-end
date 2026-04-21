from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol, Tuple

from .discovery import (
    DEFAULT_WIRE_REGISTRY_PATH,
    FileBasedAnatomyDiscovery,
    FileBasedWireRegistryDiscovery,
)
from .models import (
    AgentRef,
    AnatomyBranch,
    AorticArchAnatomy,
    CandidateSummary,
    EvaluationArtifacts,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationReport,
    PolicySpec,
    TargetModeDescriptor,
    TrialResult,
    WireRef,
)
from .runner import run_single_trial
from .runtime import PreparedEvaluationRuntime, prepare_evaluation_runtime
from .target_discovery import AnatomyTargetDiscovery


def _generated_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_mean(values: Tuple[float | None, ...]) -> Optional[float]:
    finite = tuple(
        float(value)
        for value in values
        if value is not None and float(value) == float(value)
    )
    if not finite:
        return None
    return sum(finite) / len(finite)


def _finite_std(values: Tuple[float | None, ...]) -> Optional[float]:
    finite = tuple(
        float(value)
        for value in values
        if value is not None and float(value) == float(value)
    )
    if not finite:
        return None
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return variance**0.5


def _maybe_close(obj: object) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()


def summarize_trials(trials: Tuple[TrialResult, ...]) -> CandidateSummary:
    """Aggregate all trials for one candidate/scenario pair."""

    if not trials:
        raise ValueError("trials must not be empty")

    first = trials[0]
    return CandidateSummary(
        scenario_name=first.scenario_name,
        candidate_name=first.candidate_name,
        execution_wire=first.execution_wire,
        trained_on_wire=first.policy.trained_on_wire,
        trial_count=len(trials),
        success_rate=_finite_mean(
            tuple(1.0 if trial.telemetry.success else 0.0 for trial in trials)
        ),
        score_mean=_finite_mean(tuple(trial.score.total for trial in trials)),
        score_std=_finite_std(tuple(trial.score.total for trial in trials)),
        steps_total_mean=_finite_mean(
            tuple(float(trial.telemetry.steps_total) for trial in trials)
        ),
        steps_to_success_mean=_finite_mean(
            tuple(
                None
                if trial.telemetry.steps_to_success is None
                else float(trial.telemetry.steps_to_success)
                for trial in trials
            )
        ),
        tip_speed_max_mean_mm_s=_finite_mean(
            tuple(trial.telemetry.tip_speed_max_mm_s for trial in trials)
        ),
        wall_force_max_mean=_finite_mean(
            tuple(
                None
                if trial.telemetry.forces is None
                else trial.telemetry.forces.total_force_norm_max
                for trial in trials
            )
        ),
        wall_force_max_mean_newton=_finite_mean(
            tuple(
                None
                if trial.telemetry.forces is None
                else trial.telemetry.forces.total_force_norm_max_newton
                for trial in trials
            )
        ),
        force_available_rate=_finite_mean(
            tuple(
                None
                if trial.telemetry.forces is None
                else 1.0 if trial.telemetry.forces.available_for_score else 0.0
                for trial in trials
            )
        ),
    )


class AnatomyDiscoveryPort(Protocol):
    """Discovery port for anatomy records available to eval_v2."""

    def list_anatomies(
        self,
        *,
        registry_path: Optional[Path] = None,
    ) -> Tuple[AorticArchAnatomy, ...]:
        """Return all discoverable anatomies from the configured registry file."""

    def get_anatomy(
        self,
        *,
        record_id: str,
        registry_path: Optional[Path] = None,
    ) -> AorticArchAnatomy:
        """Return one resolved anatomy by its stable registry record id."""


class PolicyDiscoveryPort(Protocol):
    """Discovery port for execution wires and evaluable policies."""

    def list_execution_wires(self) -> Tuple[WireRef, ...]:
        """Return all wires available as physical execution devices."""

    def list_startable_wires(self) -> Tuple[WireRef, ...]:
        """Return only wires with at least one loadable agent checkpoint."""

    def list_registry_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        """Return policies discoverable from the local registry-backed agents."""

    def list_explicit_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        """Return policies discoverable from explicit checkpoint sources."""

    def resolve_policy_from_agent_ref(self, agent_ref: AgentRef) -> PolicySpec:
        """Resolve one registry agent reference into a concrete policy artifact."""


class TargetDiscoveryPort(Protocol):
    """Discovery port for anatomy-specific target-selection options."""

    def list_branches(self, anatomy: AorticArchAnatomy) -> Tuple[AnatomyBranch, ...]:
        """Return all branch descriptors for one resolved anatomy."""

    def get_branch(
        self,
        anatomy: AorticArchAnatomy,
        *,
        branch_name: str,
    ) -> AnatomyBranch:
        """Return one branch descriptor for one resolved anatomy."""

    def list_target_modes(self) -> Tuple[TargetModeDescriptor, ...]:
        """Return the supported target-construction modes."""


class EvaluationRunnerPort(Protocol):
    """Execution port for fully-resolved eval_v2 jobs."""

    def run_evaluation_job(self, job: EvaluationJob) -> EvaluationReport:
        """Execute one fully-resolved evaluation job and return its report."""


class EvaluationService(ABC):
    """Single service-layer API boundary for eval_v2."""

    @abstractmethod
    def list_anatomies(
        self,
        *,
        registry_path: Optional[Path] = None,
    ) -> Tuple[AorticArchAnatomy, ...]:
        """Return discoverable anatomies as resolved domain models."""

    @abstractmethod
    def get_anatomy(
        self,
        *,
        record_id: str,
        registry_path: Optional[Path] = None,
    ) -> AorticArchAnatomy:
        """Return one anatomy selected by its stable registry record id."""

    @abstractmethod
    def list_branches(self, anatomy: AorticArchAnatomy) -> Tuple[AnatomyBranch, ...]:
        """Return branch descriptors available for target selection."""

    @abstractmethod
    def get_branch(
        self,
        anatomy: AorticArchAnatomy,
        *,
        branch_name: str,
    ) -> AnatomyBranch:
        """Return one branch descriptor by name."""

    @abstractmethod
    def list_target_modes(self) -> Tuple[TargetModeDescriptor, ...]:
        """Return target modes offered to adapters."""

    @abstractmethod
    def list_execution_wires(self) -> Tuple[WireRef, ...]:
        """Return wires that can be mounted as the execution device."""

    @abstractmethod
    def list_startable_wires(self) -> Tuple[WireRef, ...]:
        """Return wires that currently have at least one loadable agent."""

    @abstractmethod
    def list_registry_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        """Return registry-backed policies that can be offered for evaluation."""

    @abstractmethod
    def list_explicit_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        """Return non-registry policies discoverable from explicit checkpoints."""

    @abstractmethod
    def resolve_policy_from_agent_ref(self, agent_ref: AgentRef) -> PolicySpec:
        """Resolve one stable registry agent reference into a concrete policy."""

    @abstractmethod
    def build_candidate(
        self,
        *,
        name: str,
        execution_wire: WireRef,
        policy: PolicySpec,
    ) -> EvaluationCandidate:
        """Construct one candidate from a selected execution wire and policy."""

    @abstractmethod
    def list_candidates(
        self,
        *,
        execution_wire: WireRef,
        include_cross_wire: bool = True,
    ) -> Tuple[EvaluationCandidate, ...]:
        """Return discoverable candidate options for one execution wire."""

    @abstractmethod
    def run_evaluation_job(self, job: EvaluationJob) -> EvaluationReport:
        """Execute one evaluation job and return the normalized service report."""


class LocalEvaluationRunner:
    """Concrete job runner built on the clean-room eval_v2 runtime and runner."""

    def __init__(
        self,
        *,
        registry_path: Path = DEFAULT_WIRE_REGISTRY_PATH,
        runtime_factory: Callable[..., PreparedEvaluationRuntime] = prepare_evaluation_runtime,
        trial_runner: Callable[..., TrialResult] = run_single_trial,
        generated_at_factory: Callable[[], str] = _generated_at_utc,
    ) -> None:
        self._registry_path = Path(registry_path)
        self._runtime_factory = runtime_factory
        self._trial_runner = trial_runner
        self._generated_at_factory = generated_at_factory

    def run_evaluation_job(self, job: EvaluationJob) -> EvaluationReport:
        trials: list[TrialResult] = []
        for scenario in job.scenarios:
            for candidate in job.candidates:
                runtime = self._runtime_factory(
                    candidate=candidate,
                    scenario=scenario,
                    registry_path=self._registry_path,
                    policy_device=job.execution.policy_device,
                )
                try:
                    for trial_index, seed in enumerate(job.execution.seeds):
                        trial = self._trial_runner(
                            runtime=runtime,
                            trial_index=trial_index,
                            seed=seed,
                            execution=job.execution,
                            scoring=job.scoring,
                        )
                        trials.append(trial)
                finally:
                    _maybe_close(runtime.play_policy)
                    _maybe_close(runtime.intervention)

        summaries = tuple(
            summarize_trials(
                tuple(
                    trial
                    for trial in trials
                    if trial.scenario_name == scenario.name
                    and trial.candidate_name == candidate.name
                    and trial.execution_wire == candidate.execution_wire
                )
            )
            for scenario in job.scenarios
            for candidate in job.candidates
        )
        return EvaluationReport(
            job_name=job.name,
            generated_at=self._generated_at_factory(),
            summaries=summaries,
            trials=tuple(trials),
            artifacts=EvaluationArtifacts(output_dir=job.output_root / job.name),
        )


class DefaultEvaluationService(EvaluationService):
    """Default concrete service composed from file-based discovery and local execution."""

    def __init__(
        self,
        *,
        anatomy_discovery: Optional[AnatomyDiscoveryPort] = None,
        policy_discovery: Optional[PolicyDiscoveryPort] = None,
        target_discovery: Optional[TargetDiscoveryPort] = None,
        evaluation_runner: Optional[EvaluationRunnerPort] = None,
    ) -> None:
        self._anatomy_discovery = anatomy_discovery or FileBasedAnatomyDiscovery()
        self._policy_discovery = policy_discovery or FileBasedWireRegistryDiscovery()
        self._target_discovery = target_discovery or AnatomyTargetDiscovery()
        self._evaluation_runner = evaluation_runner or LocalEvaluationRunner()

    def list_anatomies(
        self,
        *,
        registry_path: Optional[Path] = None,
    ) -> Tuple[AorticArchAnatomy, ...]:
        return self._anatomy_discovery.list_anatomies(registry_path=registry_path)

    def get_anatomy(
        self,
        *,
        record_id: str,
        registry_path: Optional[Path] = None,
    ) -> AorticArchAnatomy:
        return self._anatomy_discovery.get_anatomy(
            record_id=record_id,
            registry_path=registry_path,
        )

    def list_branches(self, anatomy: AorticArchAnatomy) -> Tuple[AnatomyBranch, ...]:
        return self._target_discovery.list_branches(anatomy)

    def get_branch(
        self,
        anatomy: AorticArchAnatomy,
        *,
        branch_name: str,
    ) -> AnatomyBranch:
        return self._target_discovery.get_branch(anatomy, branch_name=branch_name)

    def list_target_modes(self) -> Tuple[TargetModeDescriptor, ...]:
        return self._target_discovery.list_target_modes()

    def list_execution_wires(self) -> Tuple[WireRef, ...]:
        return self._policy_discovery.list_execution_wires()

    def list_startable_wires(self) -> Tuple[WireRef, ...]:
        return self._policy_discovery.list_startable_wires()

    def list_registry_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        return self._policy_discovery.list_registry_policies(
            execution_wire=execution_wire
        )

    def list_explicit_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        return self._policy_discovery.list_explicit_policies(
            execution_wire=execution_wire
        )

    def resolve_policy_from_agent_ref(self, agent_ref: AgentRef) -> PolicySpec:
        return self._policy_discovery.resolve_policy_from_agent_ref(agent_ref)

    def build_candidate(
        self,
        *,
        name: str,
        execution_wire: WireRef,
        policy: PolicySpec,
    ) -> EvaluationCandidate:
        return EvaluationCandidate(
            name=name,
            execution_wire=execution_wire,
            policy=policy,
        )

    def list_candidates(
        self,
        *,
        execution_wire: WireRef,
        include_cross_wire: bool = True,
    ) -> Tuple[EvaluationCandidate, ...]:
        policies = (
            self._policy_discovery.list_registry_policies()
            + self._policy_discovery.list_explicit_policies()
        )
        if not include_cross_wire:
            policies = tuple(
                policy
                for policy in policies
                if policy.trained_on_wire is None or policy.trained_on_wire == execution_wire
            )

        candidates: list[EvaluationCandidate] = []
        used_names: dict[str, int] = {}
        for policy in policies:
            base_name = self._candidate_name(policy, execution_wire=execution_wire)
            count = used_names.get(base_name, 0)
            used_names[base_name] = count + 1
            name = base_name if count == 0 else f"{base_name} #{count + 1}"
            candidates.append(
                self.build_candidate(
                    name=name,
                    execution_wire=execution_wire,
                    policy=policy,
                )
            )
        return tuple(candidates)

    def run_evaluation_job(self, job: EvaluationJob) -> EvaluationReport:
        return self._evaluation_runner.run_evaluation_job(job)

    @staticmethod
    def _candidate_name(
        policy: PolicySpec,
        *,
        execution_wire: WireRef,
    ) -> str:
        trained_on_wire = policy.trained_on_wire
        if trained_on_wire is None or trained_on_wire == execution_wire:
            return policy.name
        return f"{policy.name} [{trained_on_wire.tool_ref} -> {execution_wire.tool_ref}]"


__all__ = [
    "AnatomyDiscoveryPort",
    "DefaultEvaluationService",
    "EvaluationRunnerPort",
    "EvaluationService",
    "LocalEvaluationRunner",
    "PolicyDiscoveryPort",
    "TargetDiscoveryPort",
    "summarize_trials",
]
