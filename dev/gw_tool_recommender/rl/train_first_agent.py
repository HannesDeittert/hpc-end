"""Minimal first RL training entrypoint using stEVE_rl (SAC).

This script lives in your code (not third_party). It:
1) loads a stored device from data/<name>/tool.py
2) builds a simple wire-navigation environment
3) trains a SAC agent via eve_rl

Run:
  python3 dev/gw_tool_recommender/rl/train_first_agent.py --tool guidewire --steps 200000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List


HERE = Path(__file__).resolve()


def _repo_root() -> Path:
    for parent in (HERE.parent, *HERE.parents):
        if (parent / ".git").exists():
            return parent
    return HERE.parents[-1]


REPO_ROOT = _repo_root()

# Ensure third_party packages are importable without installation.
sys.path.insert(0, str(REPO_ROOT / "third_party" / "stEVE_training" / "eve"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "stEVE_training" / "eve_bench"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "stEVE_rl"))

import eve  # noqa: E402
import eve_rl  # noqa: E402

from dev.gw_tool_recommender.devices import make_device  # noqa: E402


def build_env(device_name: str, seed: int = 0) -> eve.Env:
    vessel_tree = eve.intervention.vesseltree.AorticArch(seed=seed)
    device = make_device(device_name)

    simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=7.5,
        image_rot_zx=[20, 5],
    )
    target = eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=5,
        branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"],
    )
    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
        stop_device_at_tree_end=True,
        normalize_action=True,
    )

    start = eve.start.MaxDeviceLength(intervention=intervention, max_length=500)
    pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)

    position = eve.observation.Tracking2D(intervention=intervention, n_points=5)
    position = eve.observation.wrapper.NormalizeTracking2DEpisode(position, intervention)
    target_state = eve.observation.Target2D(intervention=intervention)
    target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
        target_state, intervention
    )
    rotation = eve.observation.Rotations(intervention=intervention)
    obs = eve.observation.ObsDict(
        {"position": position, "target": target_state, "rotation": rotation}
    )

    target_reward = eve.reward.TargetReached(intervention=intervention, factor=1.0)
    path_delta = eve.reward.PathLengthDelta(pathfinder=pathfinder, factor=0.01)
    reward = eve.reward.Combination([target_reward, path_delta])

    terminal = eve.terminal.TargetReached(intervention=intervention)
    truncation = eve.truncation.MaxSteps(200)

    env = eve.Env(
        intervention=intervention,
        observation=obs,
        reward=reward,
        terminal=terminal,
        truncation=truncation,
        start=start,
        visualisation=None,
    )
    return env


def train_sac(
    env: eve.Env,
    steps: int,
    hidden_layers: List[int],
    lr: float,
    gamma: float,
    batch_size: int,
    replay_size: int,
    device: str,
    out_dir: Path,
):
    obs_dict = env.observation_space.sample()
    obs_list = [obs.flatten() for obs in obs_dict.values()]
    n_observations = sum(o.shape[0] for o in obs_list)
    n_actions = env.action_space.sample().flatten().shape[0]

    q1_mlp = eve_rl.network.component.MLP(hidden_layers)
    q2_mlp = eve_rl.network.component.MLP(hidden_layers)
    pol_mlp = eve_rl.network.component.MLP(hidden_layers)

    q1 = eve_rl.network.QNetwork(q1_mlp, n_observations, n_actions)
    q2 = eve_rl.network.QNetwork(q2_mlp, n_observations, n_actions)
    policy = eve_rl.network.GaussianPolicy(pol_mlp, n_observations, n_actions)

    q1_opt = eve_rl.optim.Adam(q1, lr)
    q2_opt = eve_rl.optim.Adam(q2, lr)
    pol_opt = eve_rl.optim.Adam(policy, lr)

    sac_model = eve_rl.model.SACModel(
        q1=q1,
        q2=q2,
        policy=policy,
        q1_optimizer=q1_opt,
        q2_optimizer=q2_opt,
        policy_optimizer=pol_opt,
        lr_alpha=lr,
    )
    algo = eve_rl.algo.SAC(sac_model, n_actions=n_actions, gamma=gamma)
    replay_buffer = eve_rl.replaybuffer.VanillaStep(replay_size, batch_size)

    agent = eve_rl.agent.Single(
        algo=algo,
        env_train=env,
        env_eval=env,
        replay_buffer=replay_buffer,
        consecutive_action_steps=1,
        device=eve_rl.util.torch.device(device),
        normalize_actions=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    heatup_steps = min(5000, steps // 10)
    agent.heatup(steps=heatup_steps, custom_action_low=[-10.0, -1.5])

    explore_chunk = 1000
    while agent.step_counter.exploration < steps:
        agent.explore(steps=explore_chunk)
        update_steps = agent.step_counter.exploration - agent.step_counter.update
        agent.update(steps=update_steps)

        if agent.step_counter.exploration % 50000 < explore_chunk:
            agent.save_checkpoint(str(checkpoint_dir / f"checkpoint_{agent.step_counter.exploration}"))

    agent.save_checkpoint(str(checkpoint_dir / "checkpoint_final"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, help="device name under data/<tool>/tool.py")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--hidden", nargs="+", type=int, default=[256, 256, 256])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0", "mps"])
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "first_agent"))
    args = parser.parse_args()

    env = build_env(args.tool, seed=0)
    train_sac(
        env=env,
        steps=args.steps,
        hidden_layers=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        device=args.device,
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
