#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from steve_recommender.adapters import eve
from steve_recommender.evaluation.config import AorticArchSpec
from steve_recommender.evaluation.intervention_factory import build_aortic_arch_intervention
from steve_recommender.evaluation.torch_checkpoint_compat import (
    legacy_checkpoint_load_context,
)
from steve_recommender.rl.bench_env import BenchEnv


@dataclass
class FieldRef:
    object_path: str
    object_ref: Any
    field_name: str


def _safe_name(obj: Any, fallback: str) -> str:
    try:
        name_attr = getattr(obj, "name", None)
        if hasattr(name_attr, "value"):
            name_val = str(name_attr.value).strip()
            if name_val:
                return name_val
        if name_attr is not None:
            name_val = str(name_attr).strip()
            if name_val:
                return name_val
    except Exception:
        pass
    try:
        if hasattr(obj, "getName"):
            name_val = str(obj.getName()).strip()
            if name_val:
                return name_val
    except Exception:
        pass
    return fallback


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except Exception:
        return []


def _node_children(node: Any) -> List[Any]:
    for attr in ("children",):
        try:
            return _to_list(getattr(node, attr))
        except Exception:
            pass
    for fn_name in ("getChildren",):
        try:
            fn = getattr(node, fn_name)
            if callable(fn):
                return _to_list(fn())
        except Exception:
            pass
    return []


def _node_objects(node: Any) -> List[Any]:
    for attr in ("objects",):
        try:
            return _to_list(getattr(node, attr))
        except Exception:
            pass
    for fn_name in ("getObjects",):
        try:
            fn = getattr(node, fn_name)
            if callable(fn):
                return _to_list(fn())
        except Exception:
            pass
    return []


def _iter_scene_objects(root: Any) -> Iterator[Tuple[str, Any]]:
    stack: List[Tuple[str, Any]] = [("/root", root)]
    while stack:
        node_path, node = stack.pop()
        for obj in _node_objects(node):
            obj_name = _safe_name(obj, obj.__class__.__name__)
            yield f"{node_path}/{obj_name}", obj
        children = _node_children(node)
        for idx, child in enumerate(reversed(children)):
            child_name = _safe_name(child, f"child_{len(children)-1-idx}")
            stack.append((f"{node_path}/{child_name}", child))


def _list_data_field_names(obj: Any) -> List[str]:
    def _name_from_data_desc(desc: Any) -> Optional[str]:
        try:
            if isinstance(desc, str):
                m = re.search(r"name='([^']+)'", desc)
                if m:
                    return m.group(1)
                return desc
        except Exception:
            return None
        for attr in ("name",):
            try:
                v = getattr(desc, attr)
                if hasattr(v, "value"):
                    v = v.value
                v = str(v).strip()
                if v:
                    return v
            except Exception:
                pass
        for fn_name in ("getName",):
            try:
                fn = getattr(desc, fn_name)
                if callable(fn):
                    v = str(fn()).strip()
                    if v:
                        return v
            except Exception:
                pass
        try:
            text = str(desc)
            m = re.search(r"name='([^']+)'", text)
            if m:
                return m.group(1)
        except Exception:
            pass
        return None

    names: List[str] = []
    try:
        fields = obj.getDataFields()
        if isinstance(fields, dict):
            for k in fields.keys():
                name = _name_from_data_desc(k)
                if name:
                    names.append(name)
        else:
            for k in _to_list(fields):
                name = _name_from_data_desc(k)
                if name:
                    names.append(name)
    except Exception:
        pass

    if not names:
        for attr in dir(obj):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(obj, attr)
            except Exception:
                continue
            if callable(value):
                continue
            if hasattr(value, "value"):
                names.append(attr)
    # preserve order + uniq
    seen = set()
    out = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _read_field_value(obj: Any, field_name: str) -> Any:
    try:
        data = getattr(obj, field_name)
    except Exception:
        return None
    try:
        if hasattr(data, "value"):
            return data.value
    except Exception:
        return None
    return data


def _numeric_stats(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size == 0:
        return {"size": 0, "nnz": 0, "max_abs": 0.0, "mean_abs": 0.0}
    if arr.dtype.kind not in {"i", "u", "f", "b"}:
        try:
            arr = arr.astype(np.float64)
        except Exception:
            return None
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"size": 0, "nnz": 0, "max_abs": 0.0, "mean_abs": 0.0}
    if not np.any(np.isfinite(arr)):
        return {"size": int(arr.size), "nnz": 0, "max_abs": float("nan"), "mean_abs": float("nan")}
    finite = arr[np.isfinite(arr)]
    abs_arr = np.abs(finite)
    nnz = int(np.count_nonzero(abs_arr > 0.0))
    return {
        "size": int(arr.size),
        "nnz": nnz,
        "max_abs": float(np.max(abs_arr)) if abs_arr.size else 0.0,
        "mean_abs": float(np.mean(abs_arr)) if abs_arr.size else 0.0,
    }


def _discover_field_refs(root: Any, name_pattern: re.Pattern[str]) -> List[FieldRef]:
    refs: List[FieldRef] = []
    for object_path, obj in _iter_scene_objects(root):
        for field_name in _list_data_field_names(obj):
            if not name_pattern.search(field_name):
                continue
            refs.append(FieldRef(object_path=object_path, object_ref=obj, field_name=field_name))
    return refs


def _probe_step(field_refs: Sequence[FieldRef]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ref in field_refs:
        value = _read_field_value(ref.object_ref, ref.field_name)
        stats = _numeric_stats(value)
        if stats is None:
            continue
        out.append(
            {
                "object_path": ref.object_path,
                "field": ref.field_name,
                **stats,
            }
        )
    return out


def _print_step_header(step: int, sim: Any) -> None:
    lcp_max = float("nan")
    lcp_n = 0
    lcp_src = "missing"
    try:
        lcp_forces = np.asarray(sim.root.LCP.constraintForces.value, dtype=np.float64)
        lcp_n = int(lcp_forces.size)
        lcp_src = "/root/LCP.constraintForces"
        lcp_max = float(np.nanmax(np.abs(lcp_forces))) if lcp_forces.size else 0.0
    except Exception:
        pass
    point_contacts = 0
    line_contacts = 0
    try:
        point_contacts = int(sim._instruments_combined.CollisionModel.PointCollisionModel.numberOfContacts.value)  # noqa: SLF001
    except Exception:
        pass
    try:
        line_contacts = int(sim._instruments_combined.CollisionModel.LineCollisionModel.numberOfContacts.value)  # noqa: SLF001
    except Exception:
        pass
    print(
        f"[probe] step={step} "
        f"LCPmax={lcp_max:.5g} LCPn={lcp_n} "
        f"pointContacts={point_contacts} lineContacts={line_contacts} "
        f"LCPsrc={lcp_src}"
    )


def _lcp_from_stats(stats: Sequence[Dict[str, Any]]) -> Tuple[float, int, str]:
    lcp_entries = [
        s
        for s in stats
        if str(s.get("field", "")).lower() == "constraintforces"
    ]
    if not lcp_entries:
        return float("nan"), 0, "missing"
    max_entry = max(lcp_entries, key=lambda s: float(s.get("max_abs", 0.0)))
    return (
        float(max_entry.get("max_abs", 0.0)),
        int(max_entry.get("size", 0)),
        f"{max_entry.get('object_path', '?')}.{max_entry.get('field', '?')}",
    )


def _try_set_data(obj: Any, field_name: str, value: Any) -> bool:
    try:
        field = getattr(obj, field_name)
    except Exception:
        return False
    try:
        if hasattr(field, "value"):
            field.value = value
            return True
    except Exception:
        return False
    return False


def _enable_constraint_forces(root: Any) -> int:
    changed = 0
    for _, obj in _iter_scene_objects(root):
        for field_name in _list_data_field_names(obj):
            if field_name != "computeConstraintForces":
                continue
            if _try_set_data(obj, field_name, True):
                changed += 1
    return changed


def _build_manual_action(space, insert_val: float, rotate_val: float, rng: np.random.Generator, random_actions: bool) -> np.ndarray:
    if random_actions:
        return np.asarray(space.sample(), dtype=np.float32)
    shape = tuple(space.shape)
    low = np.asarray(space.low, dtype=np.float32)
    high = np.asarray(space.high, dtype=np.float32)
    action = np.zeros(shape, dtype=np.float32)
    if len(shape) == 1 and shape[0] >= 2:
        action[0] = insert_val
        action[1] = rotate_val
    elif len(shape) == 2 and shape[1] >= 2:
        action[:, 0] = insert_val
        action[:, 1] = rotate_val
    else:
        # fallback: small random action inside bounds
        action = rng.uniform(low=low, high=high, size=shape).astype(np.float32)
    return np.clip(action, low, high)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Step-by-step SOFA force probe: discover + inspect numeric force/contact fields."
    )
    p.add_argument("--tool", required=True, help="Wire ref, e.g. TestModel_StandardJ035/StandardJ035_PTFE")
    p.add_argument("--arch-type", default="I")
    p.add_argument("--arch-seed", type=int, default=30)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--max-episode-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--action-insert", type=float, default=0.4)
    p.add_argument("--action-rotate", type=float, default=0.0)
    p.add_argument("--random-actions", action="store_true")
    p.add_argument("--checkpoint", type=str, default="", help="Optional agent checkpoint (*.everl). If set, actions come from policy.")
    p.add_argument("--policy-device", type=str, default="cpu")
    p.add_argument("--pause", action="store_true", help="Wait for ENTER before each step.")
    p.add_argument("--field-regex", default=r"(force|constraint|contact|lcp|lambda|jacob)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--show-zero", action="store_true")
    p.add_argument("--dump-fields", action="store_true")
    p.add_argument(
        "--enable-constraint-forces",
        action="store_true",
        help="Set solver field computeConstraintForces=1 where available.",
    )
    p.add_argument("--csv-out", type=str, default="")
    p.add_argument("--visualize", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    anatomy = AorticArchSpec(
        type="aortic_arch",
        arch_type=str(args.arch_type),
        seed=int(args.arch_seed),
        target_mode="branch_end",
        target_branches=["lcca"],
        target_threshold_mm=5.0,
    )
    intervention, _ = build_aortic_arch_intervention(tool_ref=args.tool, anatomy=anatomy)
    env = BenchEnv(
        intervention=intervention,
        mode="eval",
        visualisation=bool(args.visualize),
        n_max_steps=int(args.max_episode_steps),
    )
    # Probe requires in-process SOFA access.
    intervention.make_non_mp()

    obs, info = env.reset(seed=int(args.seed))
    _ = info

    algo = None
    obs_flat = None
    if args.checkpoint:
        import torch
        from steve_recommender.adapters import eve_rl

        with legacy_checkpoint_load_context():
            algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(args.checkpoint)
        algo.to(torch.device(str(args.policy_device)))
        algo.reset()
        obs_flat, _ = eve_rl.util.flatten_obs(obs)
    sim = intervention.simulation
    if sim is None or not hasattr(sim, "root"):
        raise RuntimeError("Simulation root unavailable (non-mp simulation required).")

    name_pattern = re.compile(args.field_regex, re.IGNORECASE)
    refs = _discover_field_refs(sim.root, name_pattern)
    print(f"[probe] discovered {len(refs)} candidate data fields matching /{args.field_regex}/")
    if args.enable_constraint_forces:
        changed = _enable_constraint_forces(sim.root)
        print(f"[probe] enabled computeConstraintForces on {changed} object(s)")
    if args.dump_fields:
        for ref in refs:
            print(f"  - {ref.object_path}.{ref.field_name}")
    if not refs:
        print("[probe] no matching fields found; try a broader --field-regex")
        return

    csv_rows: List[Dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        if args.pause:
            cmd = input(f"[probe] step={step} ENTER=next, q=quit > ").strip().lower()
            if cmd == "q":
                break

        if algo is None:
            action = _build_manual_action(
                env.action_space,
                insert_val=float(args.action_insert),
                rotate_val=float(args.action_rotate),
                rng=rng,
                random_actions=bool(args.random_actions),
            )
        else:
            from steve_recommender.adapters import eve_rl

            action_model = algo.get_eval_action(obs_flat)
            action_model = np.asarray(action_model, dtype=np.float32).reshape(
                env.action_space.shape
            )
            action = (action_model + 1.0) / 2.0 * (
                env.action_space.high - env.action_space.low
            ) + env.action_space.low
            action = np.asarray(action, dtype=np.float32)

        obs, _, terminal, truncation, _ = env.step(action)
        if algo is not None:
            from steve_recommender.adapters import eve_rl

            obs_flat, _ = eve_rl.util.flatten_obs(obs)
        if args.visualize:
            env.render()

        _print_step_header(step, sim)
        stats_all = _probe_step(refs)
        lcp_max_stats, lcp_n_stats, lcp_src_stats = _lcp_from_stats(stats_all)
        if np.isfinite(lcp_max_stats):
            print(
                f"[probe] lcpFromScan max={lcp_max_stats:.5g} n={lcp_n_stats} src={lcp_src_stats}"
            )

        stats_print = stats_all
        if not args.show_zero:
            stats_print = [
                s for s in stats_all if np.isfinite(s["max_abs"]) and float(s["max_abs"]) > 0.0
            ]
        stats_sorted = sorted(
            stats_print,
            key=lambda s: (float(s["max_abs"]) if np.isfinite(s["max_abs"]) else -1.0),
            reverse=True,
        )
        for s in stats_sorted[: max(1, int(args.top_k))]:
            print(
                f"  {s['object_path']}.{s['field']}: "
                f"maxAbs={s['max_abs']:.5g} meanAbs={s['mean_abs']:.5g} "
                f"nnz={s['nnz']}/{s['size']}"
            )
        if not stats_sorted:
            print("  (no non-zero matching numeric fields)")

        for s in stats_all:
            csv_rows.append({"step": step, **s})

        if terminal or truncation:
            print(f"[probe] episode ended at step={step} terminal={terminal} truncation={truncation}")
            break

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "object_path",
                    "field",
                    "size",
                    "nnz",
                    "max_abs",
                    "mean_abs",
                ],
                delimiter=";",
            )
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        print(f"[probe] wrote CSV: {out_path}")

    if algo is not None:
        try:
            algo.close()
        except Exception:
            pass
    env.close()


if __name__ == "__main__":
    main()
