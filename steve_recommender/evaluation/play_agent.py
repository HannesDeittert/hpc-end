from __future__ import annotations

import argparse

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a trained agent on a (possibly unseen) aortic-arch anatomy with SofaPygame."
    )
    parser.add_argument(
        "--tool",
        required=True,
        help="Wire ref, e.g. TestModel_StandardJ035/StandardJ035_PTFE",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to an EveRL checkpoint (*.everl).",
    )
    parser.add_argument(
        "--arch-record",
        help=(
            "ID of an AorticArchRecord from the dataset index "
            "(e.g. arch_000123). If omitted, you must specify --arch-type and --arch-seed."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root (default: results/anatomies/aortic_arch).",
    )
    parser.add_argument(
        "--arch-type",
        help="Fallback: stEVE ArchType (e.g. I, II, IV) if no --arch-record is given.",
    )
    parser.add_argument(
        "--arch-seed",
        type=int,
        help="Fallback: seed for the AorticArch generator if no --arch-record is given.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="How many evaluation episodes to play (default: 3).",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=500,
        help="Max steps per episode before truncation (default: 500).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Policy device for the play-only algo (default: cuda).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=123,
        help="Base seed for episodes (default: 123).",
    )
    return parser.parse_args()


def _load_anatomy_spec(args) -> "AorticArchSpec":
    """Build an AorticArchSpec either from a dataset record or explicit type+seed."""

    from steve_recommender.anatomy.aortic_arch_dataset import (
        AorticArchRecord,
        load_aortic_arch_dataset,
    )
    from steve_recommender.evaluation.config import AorticArchSpec

    if args.arch_record:
        dataset = load_aortic_arch_dataset(args.dataset_root)
        record: AorticArchRecord | None = None
        for r in dataset.iter_index():
            if r.record_id == args.arch_record:
                record = r
                break
        if record is None:
            raise SystemExit(f"arch-record '{args.arch_record}' not found in {dataset.root}/index.jsonl")

        return AorticArchSpec(
            type="aortic_arch",
            arch_type=record.arch_type,
            seed=record.seed,
            rotation_yzx_deg=record.rotation_yzx_deg,
            scaling_xyzd=record.scaling_xyzd,
            omit_axis=record.omit_axis,
            target_mode="branch_end",
            target_branches=("lcca",),
            target_threshold_mm=5.0,
        )

    if not args.arch_type or args.arch_seed is None:
        raise SystemExit("Either --arch-record or (--arch-type and --arch-seed) must be provided.")

    return AorticArchSpec(
        type="aortic_arch",
        arch_type=str(args.arch_type),
        seed=int(args.arch_seed),
        rotation_yzx_deg=None,
        scaling_xyzd=None,
        omit_axis=None,
        target_mode="branch_end",
        target_branches=("lcca",),
        target_threshold_mm=5.0,
    )


def main() -> None:
    args = _parse_args()

    # Ensure stEVE / EveRL are importable.
    import torch

    from steve_recommender.evaluation.intervention_factory import build_aortic_arch_intervention
    from steve_recommender.rl.bench_env import BenchEnv
    from steve_recommender.adapters import eve, eve_rl

    anatomy_spec = _load_anatomy_spec(args)

    # Build intervention for the chosen anatomy + device.
    intervention, _ = build_aortic_arch_intervention(tool_ref=args.tool, anatomy=anatomy_spec)

    # BenchEnv with visualisation=True uses SofaPygame under the hood.
    env = BenchEnv(
        intervention=intervention,
        mode="eval",
        visualisation=True,
        n_max_steps=args.max_episode_steps,
    )

    device = torch.device(args.device)
    algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(args.checkpoint)
    algo.to(device)

    try:
        seed = int(args.base_seed)
        for ep in range(int(args.episodes)):
            print(f"[play] episode {ep + 1}/{args.episodes} seed={seed}")
            algo.reset()
            # Eve / Gymnasium-style API returns (obs, info)
            obs, _ = env.reset(seed=seed)
            # `flatten_obs` expects numpy arrays / dicts, matching training code.
            obs_flat, _ = eve_rl.util.flatten_obs(obs)

            while True:
                action = algo.get_eval_action(obs_flat)
                if not isinstance(action, np.ndarray):
                    action = np.asarray(action, dtype=np.float32)

                # Map normalized [-1, 1] actions back to the environment range,
                # mirroring eve_rl.agent.Single._play_episode.
                env_action = action.reshape(env.action_space.shape)
                env_action = (env_action + 1.0) / 2.0 * (
                    env.action_space.high - env.action_space.low
                ) + env.action_space.low

                obs, reward, terminal, truncation, info = env.step(env_action)
                obs_flat, _ = eve_rl.util.flatten_obs(obs)
                env.render()

                if terminal or truncation:
                    print(f"[play] episode done: reward={reward} terminal={terminal} truncation={truncation}")
                    break

            seed += 1
    except KeyboardInterrupt:
        print("[play] interrupted by user")
    finally:
        try:
            algo.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
