"""Register trained agents in the local wire library."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from steve_recommender.storage import parse_wire_ref, wire_agents_dir


@dataclass(frozen=True)
class AgentRegistration:
    name: str
    tool: str
    checkpoint: Optional[str]
    run_dir: Optional[str]
    created_at: str


def register_agent(
    *,
    tool_ref: str,
    checkpoint_path: Optional[Path],
    agent_name: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> Path:
    """Create/update an agent registry entry under data/<model>/wires/<wire>/agents/."""

    model, wire = parse_wire_ref(tool_ref)
    if not model:
        raise ValueError("tool_ref must be 'model/wire' to register an agent")

    agent_name = agent_name or Path(checkpoint_path).stem if checkpoint_path else "agent"
    agent_dir = wire_agents_dir(model, wire) / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)

    record = AgentRegistration(
        name=agent_name,
        tool=tool_ref,
        checkpoint=str(checkpoint_path) if checkpoint_path else None,
        run_dir=str(run_dir) if run_dir else None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    (agent_dir / "agent.json").write_text(
        json.dumps(asdict(record), indent=2),
        encoding="utf-8",
    )
    return agent_dir
