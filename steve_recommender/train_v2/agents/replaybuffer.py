"""Resumable replay-buffer implementations for train_v2."""

from __future__ import annotations

from copy import deepcopy
from time import sleep
from typing import Any, Dict

import torch

import eve_rl
from eve_rl.replaybuffer.vanillashared import (
    VanillaEpisodeShared as _VanillaEpisodeShared,
)
from eve_rl.replaybuffer.vanillashared import VanillaSharedBase as _VanillaSharedBase
from eve_rl.replaybuffer.vanillashared import VanillaStepShared as _VanillaStepShared


class ResumableVanillaEpisode(eve_rl.replaybuffer.VanillaEpisode):
    """Episode replay buffer with serializable state."""

    def state_dict(self) -> Dict[str, Any]:
        """Export replay-buffer content for checkpointing."""

        return {
            "capacity": self.capacity,
            "batch_size": self.batch_size,
            "buffer": deepcopy(self.buffer),
            "position": self.position,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore replay-buffer content from a serialized state."""

        self.capacity = int(state_dict["capacity"])
        self._batch_size = int(state_dict["batch_size"])
        self.buffer = deepcopy(state_dict["buffer"])
        self.position = int(state_dict["position"])

    def get_config_dict(self) -> Dict[str, Any]:
        """Return a config dict compatible with eve_rl serialization."""

        return {
            "_class": f"{self.__module__}.{self.__class__.__name__}",
            "_id": id(self),
            "capacity": self.capacity,
            "batch_size": self.batch_size,
        }


class ResumableVanillaStep(eve_rl.replaybuffer.VanillaStep):
    """Step replay buffer with serializable state."""

    def state_dict(self) -> Dict[str, Any]:
        """Export replay-buffer content for checkpointing."""

        return {
            "capacity": self.capacity,
            "batch_size": self.batch_size,
            "buffer": deepcopy(self.buffer),
            "position": self.position,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore replay-buffer content from a serialized state."""

        self.capacity = int(state_dict["capacity"])
        self._batch_size = int(state_dict["batch_size"])
        self.buffer = deepcopy(state_dict["buffer"])
        self.position = int(state_dict["position"])

    def get_config_dict(self) -> Dict[str, Any]:
        """Return a config dict compatible with eve_rl serialization."""

        return {
            "_class": f"{self.__module__}.{self.__class__.__name__}",
            "_id": id(self),
            "capacity": self.capacity,
            "batch_size": self.batch_size,
        }


class _SharedReplayStateMixin:
    def state_dict(self) -> Dict[str, Any]:
        if self._shutdown_event.is_set():
            return {
                "capacity": getattr(self, "capacity", 0),
                "batch_size": self.batch_size,
                "buffer": [],
                "position": 0,
            }

        with self._request_lock:
            self._task_queue.put(["state_dict"])
            state = self._result_queue.get()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self._shutdown_event.is_set():
            return

        with self._request_lock:
            self._task_queue.put(["load_state_dict", state_dict])
            self._result_queue.get()


class ResumableVanillaStepShared(_VanillaStepShared):
    """Shared step replay buffer with checkpoint support."""

    state_dict = _SharedReplayStateMixin.state_dict
    load_state_dict = _SharedReplayStateMixin.load_state_dict

    def run(self) -> None:
        """Spawn the internal replay-buffer worker loop."""

        internal_replay_buffer = ResumableVanillaStep(self.capacity, self._batch_size)
        self.loop(internal_replay_buffer)

    def loop(self, internal_replay_buffer: eve_rl.replaybuffer.ReplayBuffer) -> None:
        """Serve push/sample/checkpoint requests for the shared buffer."""

        while not self._shutdown_event.is_set():
            if (
                self._sample_queue.empty()
                and len(internal_replay_buffer) > self.batch_size
            ):
                batch = internal_replay_buffer.sample()
                if self.sample_device != torch.device("mps"):
                    batch = batch.to(self.sample_device)
                self._sample_queue.put(batch)
            elif not self._task_queue.empty():
                task = self._task_queue.get()
                if task[0] == "length":
                    self._result_queue.put(len(internal_replay_buffer))
                elif task[0] == "state_dict":
                    self._drain_push_queue(internal_replay_buffer)
                    self._result_queue.put(internal_replay_buffer.state_dict())
                elif task[0] == "load_state_dict":
                    self._clear_queue(self._push_queue)
                    self._clear_queue(self._sample_queue)
                    internal_replay_buffer.load_state_dict(task[1])
                    self._result_queue.put(True)
                elif task[0] == "shutdown":
                    break
            elif not self._push_queue.empty():
                batch = self._push_queue.get()
                internal_replay_buffer.push(batch)
            else:
                sleep(0.0001)
        internal_replay_buffer.close()

    @staticmethod
    def _clear_queue(target_queue) -> None:
        while not target_queue.empty():
            target_queue.get()

    def _drain_push_queue(
        self, internal_replay_buffer: eve_rl.replaybuffer.ReplayBuffer
    ) -> None:
        while not self._push_queue.empty():
            batch = self._push_queue.get()
            internal_replay_buffer.push(batch)

    def copy(self) -> _VanillaSharedBase:
        """Return the worker proxy used by eve_rl subprocesses."""

        return _VanillaSharedBase(
            self._push_queue,
            self._sample_queue,
            self._task_queue,
            self._result_queue,
            self._request_lock,
            self._shutdown_event,
            self.batch_size,
        )

    def close(self) -> None:
        """Shut down the shared replay-buffer process."""

        if self._process is None:
            return
        if self._process.is_alive():
            self._shutdown_event.set()
            self._task_queue.put(["shutdown"])
            self._process.join(5)
            if self._process.exitcode is None:
                self._process.kill()
                self._process.join()
        self._process.close()
        self._process = None

    def get_config_dict(self) -> Dict[str, Any]:
        """Return a config dict compatible with eve_rl serialization."""

        return {
            "_class": f"{self.__module__}.{self.__class__.__name__}",
            "_id": id(self),
            "capacity": self.capacity,
            "batch_size": self.batch_size,
            "sample_device": str(self.sample_device),
        }


class ResumableVanillaEpisodeShared(_VanillaEpisodeShared):
    """Shared episode replay buffer with checkpoint support."""

    state_dict = _SharedReplayStateMixin.state_dict
    load_state_dict = _SharedReplayStateMixin.load_state_dict

    def run(self) -> None:
        """Spawn the internal replay-buffer worker loop."""

        internal_replay_buffer = ResumableVanillaEpisode(
            self.capacity, self._batch_size
        )
        ResumableVanillaStepShared.loop(self, internal_replay_buffer)

    _clear_queue = staticmethod(ResumableVanillaStepShared._clear_queue)
    _drain_push_queue = ResumableVanillaStepShared._drain_push_queue
    copy = ResumableVanillaStepShared.copy
    close = ResumableVanillaStepShared.close
    get_config_dict = ResumableVanillaStepShared.get_config_dict
