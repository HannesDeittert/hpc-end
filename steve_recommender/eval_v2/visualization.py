from __future__ import annotations

from importlib import import_module
from typing import Callable, Optional, Protocol, TYPE_CHECKING

from .models import ExecutionPlan, VisualizationSpec

if TYPE_CHECKING:
    from .runtime import PreparedEvaluationRuntime


class TrialVisualisation(Protocol):
    """Minimal visualisation protocol consumed by the eval_v2 runner."""

    def reset(self, episode_nr: int = 0) -> None:
        ...

    def render(self):
        ...

    def close(self) -> None:
        ...


VisualisationFactory = Callable[[object], TrialVisualisation]


def should_visualize_trial(
    visualization: VisualizationSpec,
    *,
    trial_index: int,
) -> bool:
    """Return whether one candidate trial should be rendered."""

    return visualization.enabled and trial_index < visualization.rendered_trials_per_candidate


def _load_sofa_pygame_factory() -> VisualisationFactory:
    visualisation_module = import_module("third_party.stEVE.eve.visualisation")
    return getattr(visualisation_module, "SofaPygame")


def build_trial_visualisation(
    runtime: "PreparedEvaluationRuntime",
    *,
    execution: ExecutionPlan,
    trial_index: int,
    visualisation_factory: Optional[VisualisationFactory] = None,
) -> Optional[TrialVisualisation]:
    """Build one trial viewer from upstream stEVE when rendering is enabled."""

    visualization = execution.visualization
    if not should_visualize_trial(visualization, trial_index=trial_index):
        return None
    if visualization.force_debug_overlay:
        raise NotImplementedError(
            "force_debug_overlay is not implemented in eval_v2 yet"
        )

    factory = visualisation_factory or _load_sofa_pygame_factory()
    return factory(runtime.intervention)


__all__ = ["TrialVisualisation", "build_trial_visualisation", "should_visualize_trial"]
