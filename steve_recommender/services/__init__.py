"""Service-layer APIs used by CLI and UI."""

from .agent_service import register_agent
from .comparison_service import (
    comparison_from_dict,
    load_comparison,
    resolve_candidates as resolve_comparison_candidates,
    run_comparison,
)
from .evaluation_service import (
    evaluation_config_from_dict,
    load_evaluation_config,
    run_evaluation,
)
from .library_service import (
    list_agent_ref_metadata,
    list_agent_refs,
    list_agents,
    list_models,
    list_wires,
)
from .run_service import list_evaluation_runs, list_training_runs, tail_text
from .training_service import run_training

__all__ = [
    "comparison_from_dict",
    "evaluation_config_from_dict",
    "load_comparison",
    "list_agent_ref_metadata",
    "list_agent_refs",
    "list_agents",
    "list_evaluation_runs",
    "list_models",
    "list_training_runs",
    "list_wires",
    "resolve_comparison_candidates",
    "load_evaluation_config",
    "register_agent",
    "run_comparison",
    "run_evaluation",
    "run_training",
    "tail_text",
]
