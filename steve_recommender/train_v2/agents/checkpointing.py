"""Checkpoint and replay-buffer persistence helpers for train_v2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


LATEST_REPLAY_BUFFER_NAME = "latest_replay_buffer.everl"


def checkpoint_map_location(device: torch.device) -> torch.device:
    """Pick a safe load device for a checkpoint."""

    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def load_trusted_checkpoint(file_path: Union[str, Path], map_location: torch.device):
    """Load one locally produced checkpoint, including non-tensor payloads."""

    try:
        return torch.load(file_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(file_path, map_location=map_location)


class ReplayBufferCheckpointMixin:
    """Mixin implementing checkpoint + replay-buffer sidecar persistence."""

    embed_replay_buffer_state_in_checkpoint: bool = True
    latest_replay_buffer_path: Optional[Path] = None
    resume_replay_buffer_path: Optional[Path] = None

    def configure_replay_buffer_checkpointing(
        self,
        *,
        latest_replay_buffer_path: Optional[Union[str, Path]] = None,
        resume_replay_buffer_path: Optional[Union[str, Path]] = None,
        embed_in_checkpoint: bool = True,
    ) -> None:
        """Configure replay-buffer sidecar persistence for one agent."""

        self.embed_replay_buffer_state_in_checkpoint = embed_in_checkpoint
        self.latest_replay_buffer_path = (
            Path(latest_replay_buffer_path)
            if latest_replay_buffer_path is not None
            else None
        )
        self.resume_replay_buffer_path = (
            Path(resume_replay_buffer_path)
            if resume_replay_buffer_path is not None
            else None
        )

    def save_checkpoint(
        self,
        file_path: Union[str, Path],
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save the agent checkpoint and optional replay-buffer sidecar."""

        checkpoint_dict = {
            "algo": self.algo.get_config_dict(),
            "replay_buffer": self.replay_buffer.get_config_dict(),
            "env_train": (
                self.env_train.get_config_dict()
                if hasattr(self.env_train, "get_config_dict")
                else None
            ),
            "env_eval": (
                self.env_eval.get_config_dict()
                if hasattr(self.env_eval, "get_config_dict")
                else None
            ),
            "steps": {
                "heatup": self.step_counter.heatup,
                "exploration": self.step_counter.exploration,
                "update": self.step_counter.update,
                "evaluation": self.step_counter.evaluation,
            },
            "episodes": {
                "heatup": self.episode_counter.heatup,
                "exploration": self.episode_counter.exploration,
                "evaluation": self.episode_counter.evaluation,
            },
            "network_state_dicts": self.algo.state_dicts_network(),
            "optimizer_state_dicts": self.algo.state_dicts_optimizer(),
            "scheduler_state_dicts": self.algo.state_dicts_scheduler(),
            "additional_info": additional_info,
        }

        if self.embed_replay_buffer_state_in_checkpoint:
            checkpoint_dict["replay_buffer_state"] = self.replay_buffer.state_dict()

        if self.latest_replay_buffer_path is not None:
            checkpoint_dict["replay_buffer_sidecar"] = (
                self.latest_replay_buffer_path.name
            )
            self.save_replay_buffer_state(self.latest_replay_buffer_path)

        torch.save(checkpoint_dict, file_path)

    def save_replay_buffer_state(self, file_path: Union[str, Path]) -> None:
        """Write one standalone replay-buffer sidecar file atomically."""

        if not hasattr(self.replay_buffer, "state_dict"):
            raise TypeError("Replay buffer does not support state export")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        payload = {
            "replay_buffer_state": self.replay_buffer.state_dict(),
            "steps": {
                "heatup": self.step_counter.heatup,
                "exploration": self.step_counter.exploration,
                "update": self.step_counter.update,
                "evaluation": self.step_counter.evaluation,
            },
            "episodes": {
                "heatup": self.episode_counter.heatup,
                "exploration": self.episode_counter.exploration,
                "evaluation": self.episode_counter.evaluation,
            },
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def _load_replay_buffer_state(self, state: Dict[str, Any]) -> None:
        if not hasattr(self.replay_buffer, "load_state_dict"):
            raise TypeError("Replay buffer does not support state restoration")
        self.replay_buffer.load_state_dict(state)

    def _load_replay_buffer_sidecar(
        self, checkpoint_path: Optional[Path], checkpoint: Dict[str, Any]
    ) -> Optional[Path]:
        candidates = []
        if self.resume_replay_buffer_path is not None:
            candidates.append(self.resume_replay_buffer_path)

        sidecar = checkpoint.get("replay_buffer_sidecar")
        if sidecar and checkpoint_path is not None:
            sidecar_path = Path(sidecar)
            if not sidecar_path.is_absolute():
                sidecar_path = checkpoint_path.parent / sidecar_path
            candidates.append(sidecar_path)

        if checkpoint_path is not None:
            candidates.append(checkpoint_path.parent / LATEST_REPLAY_BUFFER_NAME)

        for candidate in candidates:
            if candidate.is_file():
                payload = load_trusted_checkpoint(
                    candidate, map_location=torch.device("cpu")
                )
                state = payload.get("replay_buffer_state", payload)
                self._load_replay_buffer_state(state)
                return candidate
        return None

    def _load_checkpoint_state(
        self, checkpoint: Dict[str, Any], checkpoint_path: Optional[Path] = None
    ) -> None:
        """Restore optimizer, network, counter, and replay-buffer state."""

        self.algo.load_state_dicts_network(checkpoint["network_state_dicts"])
        if "optimizer_state_dicts" in checkpoint:
            self.algo.load_state_dicts_optimizer(checkpoint["optimizer_state_dicts"])
        if "scheduler_state_dicts" in checkpoint:
            self.algo.load_state_dicts_scheduler(checkpoint["scheduler_state_dicts"])
        if "replay_buffer_state" in checkpoint:
            self._load_replay_buffer_state(checkpoint["replay_buffer_state"])
        else:
            self._load_replay_buffer_sidecar(checkpoint_path, checkpoint)

        self.step_counter.heatup = checkpoint["steps"]["heatup"]
        self.step_counter.exploration = checkpoint["steps"]["exploration"]
        self.step_counter.evaluation = checkpoint["steps"]["evaluation"]
        self.step_counter.update = checkpoint["steps"]["update"]

        self.episode_counter.heatup = checkpoint["episodes"]["heatup"]
        self.episode_counter.exploration = checkpoint["episodes"]["exploration"]
        self.episode_counter.evaluation = checkpoint["episodes"]["evaluation"]
