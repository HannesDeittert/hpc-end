from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from steve_recommender.comparison.config import (
    ComparisonCandidateSpec,
    ComparisonConfig,
    ResolvedCandidate,
)
from steve_recommender.evaluation.config import AgentSpec, EvaluationConfig
from steve_recommender.evaluation.pipeline import run_evaluation
from steve_recommender.storage import wire_agents_dir


_CHECKPOINT_NUM_RE = re.compile(r"^checkpoint(?P<step>\d+)\.everl$")


def _split_agent_ref(agent_ref: str) -> Tuple[str, str, str]:
    """Parse `model/version:agent` into `(model, version, agent)`."""

    if ":" not in agent_ref:
        raise ValueError(
            f"Invalid agent_ref '{agent_ref}'. Expected format: model/version:agent"
        )
    tool_ref, agent_name = agent_ref.rsplit(":", 1)
    if "/" not in tool_ref:
        raise ValueError(
            f"Invalid agent_ref '{agent_ref}'. Missing model/version segment."
        )
    model, wire = tool_ref.split("/", 1)
    model = model.strip()
    wire = wire.strip()
    agent_name = agent_name.strip()
    if not model or not wire or not agent_name:
        raise ValueError(
            f"Invalid agent_ref '{agent_ref}'. Expected format: model/version:agent"
        )
    return model, wire, agent_name


def _try_parse_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_existing_path(
    path_str: str,
    *,
    base_dirs: Optional[Iterable[Path]] = None,
) -> Optional[Path]:
    candidate = Path(path_str).expanduser()
    paths_to_try: List[Path] = []
    if candidate.is_absolute():
        paths_to_try.append(candidate)
    else:
        for base in base_dirs or ():
            paths_to_try.append((base / candidate).resolve())
        paths_to_try.append(candidate.resolve())

    for p in paths_to_try:
        if p.exists() and p.is_file():
            return p
    return None


def _candidate_checkpoint_dirs(agent_dir: Path, metadata: Dict[str, object]) -> List[Path]:
    dirs: List[Path] = [agent_dir, agent_dir / "checkpoints"]

    run_dir = metadata.get("run_dir")
    if isinstance(run_dir, str) and run_dir.strip():
        dirs.append(Path(run_dir).expanduser())

    # Keep order stable while deduplicating.
    deduped: List[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)
    return deduped


def _highest_numeric_checkpoint(checkpoint_dirs: Iterable[Path]) -> Optional[Path]:
    numbered: List[Tuple[int, Path]] = []
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            continue
        for p in checkpoint_dir.glob("checkpoint*.everl"):
            m = _CHECKPOINT_NUM_RE.match(p.name)
            if not m:
                continue
            numbered.append((int(m.group("step")), p))
    if not numbered:
        return None
    return max(numbered, key=lambda t: t[0])[1]


def _latest_everl(checkpoint_dirs: Iterable[Path]) -> Optional[Path]:
    all_everl: List[Path] = []
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            continue
        all_everl.extend(checkpoint_dir.glob("*.everl"))
    all_everl.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return all_everl[0] if all_everl else None


def _resolve_registry_candidate(spec: ComparisonCandidateSpec) -> ResolvedCandidate:
    assert spec.agent_ref
    model, wire, agent_name = _split_agent_ref(spec.agent_ref)
    agents_root = wire_agents_dir(model, wire)
    agent_dir = agents_root / agent_name
    if not agent_dir.exists():
        available = []
        if agents_root.exists():
            available = sorted(
                child.name
                for child in agents_root.iterdir()
                if child.is_dir()
                and not child.name.startswith("__")
                and (child / "agent.json").exists()
            )
        available_hint = (
            f" Available agents for '{model}/{wire}': {', '.join(available)}."
            if available
            else f" No agents found under '{agents_root}'."
        )
        raise FileNotFoundError(
            f"Agent directory not found for '{spec.agent_ref}': {agent_dir}.{available_hint}"
        )

    metadata = _try_parse_json(agent_dir / "agent.json")
    checkpoint_dirs = _candidate_checkpoint_dirs(agent_dir, metadata)

    tool_from_metadata = metadata.get("tool")
    if isinstance(tool_from_metadata, str) and tool_from_metadata.strip():
        tool_ref = tool_from_metadata.strip()
    else:
        tool_ref = f"{model}/{wire}"

    # Pinned + fallback policy.
    # 1) checkpoint_override (run config)
    if spec.checkpoint_override:
        resolved = _resolve_existing_path(
            spec.checkpoint_override,
            base_dirs=checkpoint_dirs,
        )
        if resolved is None:
            raise FileNotFoundError(
                f"checkpoint_override does not exist for '{spec.agent_ref}': {spec.checkpoint_override}"
            )
        source = "checkpoint_override"
    else:
        resolved = None
        source = ""

    # 2) explicit checkpoint in candidate (optional convenience)
    if resolved is None and spec.checkpoint:
        resolved = _resolve_existing_path(
            spec.checkpoint,
            base_dirs=checkpoint_dirs,
        )
        if resolved is None:
            raise FileNotFoundError(
                f"checkpoint does not exist for '{spec.agent_ref}': {spec.checkpoint}"
            )
        source = "candidate.checkpoint"

    # 3) preferred checkpoint from metadata.
    if resolved is None:
        preferred = metadata.get("preferred_checkpoint")
        if not preferred:
            preferred = metadata.get("checkpoint")
        if isinstance(preferred, str) and preferred.strip():
            resolved = _resolve_existing_path(
                preferred,
                base_dirs=checkpoint_dirs,
            )
            if resolved is not None:
                source = "agent.json"

    # 4) best_checkpoint.everl
    if resolved is None:
        for checkpoint_dir in checkpoint_dirs:
            best = checkpoint_dir / "best_checkpoint.everl"
            if best.exists():
                resolved = best
                source = "best_checkpoint.everl"
                break

    # 5) highest checkpoint*.everl (numeric)
    if resolved is None:
        highest = _highest_numeric_checkpoint(checkpoint_dirs)
        if highest is not None:
            resolved = highest
            source = "highest_checkpointN"

    # 6) latest *.everl
    if resolved is None:
        latest = _latest_everl(checkpoint_dirs)
        if latest is not None:
            resolved = latest
            source = "latest_everl"

    if resolved is None:
        raise FileNotFoundError(
            f"No checkpoint found for '{spec.agent_ref}' in {agent_dir}"
        )

    return ResolvedCandidate(
        name=(spec.name or agent_name),
        tool=tool_ref,
        checkpoint=resolved.resolve(),
        agent_ref=spec.agent_ref,
        source=source,
    )


def _resolve_explicit_candidate(spec: ComparisonCandidateSpec) -> ResolvedCandidate:
    if not spec.tool:
        raise ValueError(
            "Explicit comparison candidate must define 'tool' when agent_ref is not set."
        )
    if not spec.checkpoint and not spec.checkpoint_override:
        raise ValueError(
            f"Explicit candidate '{spec.name or spec.tool}' must define checkpoint or checkpoint_override."
        )

    if spec.checkpoint_override:
        resolved = _resolve_existing_path(spec.checkpoint_override, base_dirs=())
        if resolved is None:
            raise FileNotFoundError(
                f"checkpoint_override does not exist: {spec.checkpoint_override}"
            )
        source = "checkpoint_override"
    else:
        assert spec.checkpoint
        resolved = _resolve_existing_path(spec.checkpoint, base_dirs=())
        if resolved is None:
            raise FileNotFoundError(f"checkpoint does not exist: {spec.checkpoint}")
        source = "explicit_checkpoint"

    return ResolvedCandidate(
        name=(spec.name or spec.tool),
        tool=spec.tool,
        checkpoint=resolved.resolve(),
        agent_ref=None,
        source=source,
    )


def _make_unique_names(candidates: Iterable[ResolvedCandidate]) -> List[ResolvedCandidate]:
    seen: Dict[str, int] = {}
    out: List[ResolvedCandidate] = []
    for c in candidates:
        count = seen.get(c.name, 0)
        seen[c.name] = count + 1
        if count == 0:
            out.append(c)
            continue
        out.append(
            ResolvedCandidate(
                name=f"{c.name}_{count}",
                tool=c.tool,
                checkpoint=c.checkpoint,
                agent_ref=c.agent_ref,
                source=c.source,
            )
        )
    return out


def resolve_candidates(cfg: ComparisonConfig) -> List[ResolvedCandidate]:
    resolved: List[ResolvedCandidate] = []
    for spec in cfg.candidates:
        if spec.agent_ref:
            resolved.append(_resolve_registry_candidate(spec))
        else:
            resolved.append(_resolve_explicit_candidate(spec))
    return _make_unique_names(resolved)


def _to_evaluation_config(
    cfg: ComparisonConfig,
    resolved: List[ResolvedCandidate],
) -> EvaluationConfig:
    agents = [
        AgentSpec(name=c.name, tool=c.tool, checkpoint=str(c.checkpoint))
        for c in resolved
    ]
    return EvaluationConfig(
        name=cfg.name,
        agents=agents,
        anatomy=cfg.anatomy,
        n_trials=cfg.n_trials,
        base_seed=cfg.base_seed,
        seeds=cfg.seeds,
        max_episode_steps=cfg.max_episode_steps,
        output_root=cfg.output_root,
        policy_device=cfg.policy_device,
        use_non_mp_sim=cfg.use_non_mp_sim,
        stochastic_eval=cfg.stochastic_eval,
        visualize=cfg.visualize,
        visualize_trials_per_agent=cfg.visualize_trials_per_agent,
        visualize_force_debug=cfg.visualize_force_debug,
        visualize_force_debug_top_k=cfg.visualize_force_debug_top_k,
        scoring=cfg.scoring,
        force_extraction=cfg.force_extraction,
    )


def run_comparison(cfg: ComparisonConfig) -> Path:
    """Resolve candidates and execute a standard evaluation run."""

    resolved = resolve_candidates(cfg)
    eval_cfg = _to_evaluation_config(cfg, resolved)
    return run_evaluation(eval_cfg)


def resolved_candidates_to_dicts(resolved: List[ResolvedCandidate]) -> List[Dict[str, object]]:
    return [
        {
            "name": c.name,
            "tool": c.tool,
            "checkpoint": str(c.checkpoint),
            "agent_ref": c.agent_ref,
            "source": c.source,
        }
        for c in resolved
    ]


def comparison_config_to_dict(cfg: ComparisonConfig) -> Dict[str, object]:
    return asdict(cfg)
