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


VisualisationFactory = Callable[..., TrialVisualisation]


def _load_sofa_pygame_class():
    visualisation_module = import_module("third_party.stEVE.eve.visualisation")
    return getattr(visualisation_module, "SofaPygame")


SofaPygame = _load_sofa_pygame_class()


class HiddenSofaPygame(SofaPygame):
    """SofaPygame variant that keeps the SDL window hidden.

    We avoid patching vendored stEVE code by injecting the pygame.HIDDEN flag
    only while the base-class `reset()` initializes the display.
    """

    def reset(self, episode_nr: int = 0) -> None:
        if getattr(self, "_initialized", False):
            super().reset(episode_nr)
            return

        pygame_module = import_module("pygame")
        hidden_flag = int(getattr(pygame_module, "HIDDEN", 0))
        display = pygame_module.display
        original_set_mode = display.set_mode

        def _set_mode_hidden(size, flags=0, *args, **kwargs):
            return original_set_mode(size, int(flags) | hidden_flag, *args, **kwargs)

        display.set_mode = _set_mode_hidden
        try:
            super().reset(episode_nr)
        finally:
            display.set_mode = original_set_mode


def should_visualize_trial(
    visualization: VisualizationSpec,
    *,
    trial_index: int,
) -> bool:
    """Return whether one candidate trial should be rendered."""

    return visualization.enabled and trial_index < visualization.rendered_trials_per_candidate


def build_trial_visualisation(
    runtime: "PreparedEvaluationRuntime",
    *,
    execution: ExecutionPlan,
    trial_index: int,
    visualisation_factory: Optional[VisualisationFactory] = None,
    hidden_window: bool = True,
) -> Optional[TrialVisualisation]:
    """Build one trial viewer from upstream stEVE when rendering is enabled.

    `hidden_window=True` is intended for frame-streaming integrations (Qt UI),
    while CLI usage should pass `hidden_window=False` to show a normal
    interactive pygame window.
    """

    visualization = execution.visualization
    if not should_visualize_trial(visualization, trial_index=trial_index):
        return None
    if visualization.force_debug_overlay:
        raise NotImplementedError(
            "force_debug_overlay is not implemented in eval_v2 yet"
        )

    if visualisation_factory is not None:
        return visualisation_factory(runtime.intervention)

    visualisation_cls = HiddenSofaPygame if hidden_window else SofaPygame
    return visualisation_cls(runtime.intervention)


__all__ = [
    "HiddenSofaPygame",
    "SofaPygame",
    "TrialVisualisation",
    "build_trial_visualisation",
    "should_visualize_trial",
]
