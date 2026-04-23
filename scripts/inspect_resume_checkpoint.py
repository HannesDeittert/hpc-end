#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


def _tensor_stats(state_dict: Dict[str, Any]) -> Tuple[float, float, int]:
    abs_sum = 0.0
    sq_sum = 0.0
    n = 0
    for value in state_dict.values():
        if not torch.is_tensor(value):
            continue
        v = value.detach().float().reshape(-1)
        if v.numel() == 0:
            continue
        abs_sum += float(v.abs().sum().item())
        sq_sum += float((v * v).sum().item())
        n += int(v.numel())
    l2 = sq_sum ** 0.5
    return abs_sum, l2, n


def _flatten_network_tensors(network_state: Dict[str, Any]) -> Iterable[Tuple[str, torch.Tensor]]:
    for section_name, section_state in network_state.items():
        if section_name == "log_alpha":
            if torch.is_tensor(section_state):
                yield "log_alpha", section_state
            continue
        if not isinstance(section_state, dict):
            continue
        for name, tensor in section_state.items():
            if torch.is_tensor(tensor):
                yield f"{section_name}.{name}", tensor


def _network_delta(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, float, int]:
    b_lookup = {name: tensor for name, tensor in _flatten_network_tensors(b)}
    abs_delta_sum = 0.0
    sq_delta_sum = 0.0
    n = 0
    for name, a_tensor in _flatten_network_tensors(a):
        b_tensor = b_lookup.get(name)
        if b_tensor is None:
            continue
        delta = a_tensor.detach().float() - b_tensor.detach().float()
        flat = delta.reshape(-1)
        if flat.numel() == 0:
            continue
        abs_delta_sum += float(flat.abs().sum().item())
        sq_delta_sum += float((flat * flat).sum().item())
        n += int(flat.numel())
    l2_delta = sq_delta_sum ** 0.5
    return abs_delta_sum, l2_delta, n


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _print_single(path: Path) -> int:
    cp = _load_checkpoint(path)
    print(f"checkpoint: {path}")
    print(f"keys: {sorted(cp.keys())}")

    steps = cp.get("steps", {})
    episodes = cp.get("episodes", {})
    print(
        "steps: heatup={heatup} exploration={exploration} update={update} evaluation={evaluation}".format(
            heatup=steps.get("heatup"),
            exploration=steps.get("exploration"),
            update=steps.get("update"),
            evaluation=steps.get("evaluation"),
        )
    )
    print(
        "episodes: heatup={heatup} exploration={exploration} evaluation={evaluation}".format(
            heatup=episodes.get("heatup"),
            exploration=episodes.get("exploration"),
            evaluation=episodes.get("evaluation"),
        )
    )

    replay_state = cp.get("replay_buffer_state")
    has_replay_state = isinstance(replay_state, dict)
    print(f"replay_buffer_state_present: {has_replay_state}")
    if has_replay_state:
        buffer_len = len(replay_state.get("buffer", []))
        print(
            "replay_state: capacity={capacity} batch_size={batch_size} position={position} buffer_len={buffer_len}".format(
                capacity=replay_state.get("capacity"),
                batch_size=replay_state.get("batch_size"),
                position=replay_state.get("position"),
                buffer_len=buffer_len,
            )
        )

    optim = cp.get("optimizer_state_dicts")
    if isinstance(optim, dict):
        print("optimizer_state:")
        for name, state in sorted(optim.items()):
            if not isinstance(state, dict):
                continue
            print(
                f"  {name}: state_entries={len(state.get('state', {}))} param_groups={len(state.get('param_groups', []))}"
            )

    sched = cp.get("scheduler_state_dicts")
    if isinstance(sched, dict):
        print("scheduler_state:")
        for name, state in sorted(sched.items()):
            if not isinstance(state, dict):
                continue
            print(
                "  {name}: last_epoch={last_epoch} _step_count={step_count}".format(
                    name=name,
                    last_epoch=state.get("last_epoch"),
                    step_count=state.get("_step_count"),
                )
            )

    net = cp.get("network_state_dicts")
    if isinstance(net, dict):
        abs_sum, l2, n = _tensor_stats(
            {
                key: value
                for key, value in net.items()
                if torch.is_tensor(value)
            }
        )
        sections = [k for k, v in net.items() if isinstance(v, dict)]
        print(
            f"network_state_sections: {sections} | direct_tensors_n={n} direct_tensors_abs_sum={abs_sum:.6f} direct_tensors_l2={l2:.6f}"
        )

    return 0


def _print_compare(before_path: Path, after_path: Path) -> int:
    before = _load_checkpoint(before_path)
    after = _load_checkpoint(after_path)
    print(f"before: {before_path}")
    print(f"after:  {after_path}")

    before_steps = before.get("steps", {})
    after_steps = after.get("steps", {})
    for key in ("heatup", "exploration", "update", "evaluation"):
        b = int(before_steps.get(key, 0))
        a = int(after_steps.get(key, 0))
        print(f"steps.{key}: {b} -> {a} (delta={a-b})")

    before_episodes = before.get("episodes", {})
    after_episodes = after.get("episodes", {})
    for key in ("heatup", "exploration", "evaluation"):
        b = int(before_episodes.get(key, 0))
        a = int(after_episodes.get(key, 0))
        print(f"episodes.{key}: {b} -> {a} (delta={a-b})")

    before_replay = before.get("replay_buffer_state")
    after_replay = after.get("replay_buffer_state")
    if isinstance(before_replay, dict) and isinstance(after_replay, dict):
        b_len = len(before_replay.get("buffer", []))
        a_len = len(after_replay.get("buffer", []))
        print(f"replay.buffer_len: {b_len} -> {a_len} (delta={a_len-b_len})")
        print(
            f"replay.position: {before_replay.get('position')} -> {after_replay.get('position')}"
        )
    else:
        print(
            "replay_buffer_state: missing in at least one checkpoint (cannot compare buffer progression)"
        )

    before_net = before.get("network_state_dicts")
    after_net = after.get("network_state_dicts")
    if isinstance(before_net, dict) and isinstance(after_net, dict):
        abs_delta_sum, l2_delta, n = _network_delta(after_net, before_net)
        print(
            "network_delta: params_compared={n} abs_delta_sum={abs_delta_sum:.6f} l2_delta={l2_delta:.6f}".format(
                n=n,
                abs_delta_sum=abs_delta_sum,
                l2_delta=l2_delta,
            )
        )
    else:
        print("network_delta: unavailable")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect EveRL checkpoints for resume-critical state, or compare two checkpoints."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Checkpoint to inspect (.everl)",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="Optional second checkpoint to compare against (after).",
    )
    args = parser.parse_args()

    if args.compare_to is None:
        return _print_single(args.checkpoint)
    return _print_compare(args.checkpoint, args.compare_to)


if __name__ == "__main__":
    raise SystemExit(main())
