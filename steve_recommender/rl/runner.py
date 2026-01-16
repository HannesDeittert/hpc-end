from __future__ import annotations

import csv
import logging
import os
from math import inf
from typing import Any, List, Optional, Tuple

from steve_recommender.steve_adapter import eve_rl

Agent = eve_rl.agent.agent.Agent
EpisodeCounter = eve_rl.agent.agent.EpisodeCounter
StepCounter = eve_rl.agent.agent.StepCounter
EveRLObject = eve_rl.util.EveRLObject


class Runner(EveRLObject):
    """Training loop helper (stEVE_rl Runner with small robustness fixes).

    This is intentionally kept outside upstream stEVE repos so we can keep
    dependencies clean while still matching the behavior of `stEVE_training` scripts.
    """

    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        quality_info: Optional[str] = None,
        info_results: Optional[List[str]] = None,
        tensorboard_logdir: Optional[str] = None,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_for_result_file = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.quality_info = quality_info
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)

        self._tb_writer = None
        if tensorboard_logdir:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                os.makedirs(tensorboard_logdir, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=tensorboard_logdir)
                self.logger.info("TensorBoard enabled: %s", tensorboard_logdir)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "TensorBoard disabled (failed to initialize at %s): %s",
                    tensorboard_logdir,
                    exc,
                )

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        self._results["quality"] = 0.0
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best quality"] = 0.0
        self._results["best explore steps"] = 0.0

        file_existed = os.path.isfile(results_file)

        with open(results_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_existed:
                writer.writerow(
                    list(self._results.keys())
                    + [" "]
                    + list(agent_parameter_for_result_file.keys())
                )
                writer.writerow(
                    [" "] * (len(self._results.values()) + 1)
                    + list(agent_parameter_for_result_file.values())
                )

        self.best_eval = {"steps": 0, "quality": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def close(self) -> None:
        if self._tb_writer is not None:
            try:
                self._tb_writer.flush()
                self._tb_writer.close()
            finally:
                self._tb_writer = None

    def heatup(self, steps: int) -> None:
        self.agent.heatup(
            steps=steps,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
        )

    def explore(self, n_episodes: int) -> None:
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int) -> Any:
        # Agent.update uses keyword-only arguments in our stEVE_rl version.
        update_result = self.agent.update(steps=n_steps)
        self._tb_log_update(
            explore_steps=int(self.step_counter.exploration),
            update_result=update_result,
        )
        return update_result

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ) -> Tuple[float, float]:
        explore_steps = self.step_counter.exploration
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{explore_steps}.everl"
        )
        result_episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)
        if not result_episodes:
            self.logger.warning(
                "Evaluation returned 0 episodes; saving checkpoint without eval results "
                f"(episodes={episodes}, seeds={'set' if seeds else None})."
            )
            self._results["episodes explore"] = self.episode_counter.exploration
            self._results["steps explore"] = explore_steps
            self._results["reward"] = float("nan")
            self._results["quality"] = float("nan")
            for info_result_name in self.info_results:
                self._results[info_result_name] = float("nan")
            self._results["best quality"] = self.best_eval["quality"]
            self._results["best explore steps"] = self.best_eval["steps"]

            eval_results = {"episodes": []}
            eval_results.update(self._results)
            eval_results.pop("best quality")
            eval_results.pop("best explore steps")

            self.agent.save_checkpoint(checkpoint_file, eval_results)
            with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                writer.writerow(self._results.values())

            self._tb_log_eval(
                explore_steps=explore_steps,
                reward=float("nan"),
                quality=float("nan"),
                info_results={name: float("nan") for name in self.info_results},
            )
            return float("nan"), float("nan")

        qualities, rewards = [], []
        results_for_info = {name: [] for name in self.info_results}
        eval_results = {"episodes": []}
        for episode in result_episodes:
            reward = episode.episode_reward
            quality = (
                episode.infos[-1][self.quality_info]
                if self.quality_info is not None
                else reward
            )
            qualities.append(quality)
            rewards.append(reward)
            for info_result_name in self.info_results:
                info = episode.infos[-1][info_result_name]
                results_for_info[info_result_name].append(info)
            eval_results["episodes"].append(
                {
                    "seed": episode.seed,
                    "options": episode.options,
                    "quality": episode.infos[-1][self.quality_info],
                }
            )

        reward = sum(rewards) / len(rewards)
        quality = sum(qualities) / len(qualities)
        for info_result_name, results in results_for_info.items():
            result = sum(results) / len(results)
            self._results[info_result_name] = round(result, 3)
        save_best = False
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        self._results["reward"] = round(reward, 3)
        self._results["quality"] = round(quality, 3)
        self._results["best quality"] = self.best_eval["quality"]
        self._results["best explore steps"] = self.best_eval["steps"]

        eval_results.update(self._results)
        eval_results.pop("best quality")
        eval_results.pop("best explore steps")

        self.agent.save_checkpoint(checkpoint_file, eval_results)
        if save_best:
            checkpoint_file = os.path.join(self.checkpoint_folder, "best_checkpoint.everl")
            self.agent.save_checkpoint(checkpoint_file, eval_results)

        self._tb_log_eval(
            explore_steps=explore_steps,
            reward=reward,
            quality=quality,
            info_results={name: self._results[name] for name in self.info_results},
        )
        self.logger.info(
            f"Quality: {quality}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        return quality, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ) -> None:
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            update_steps = (
                self.step_counter.exploration * update_steps_per_explore_step
                - self.step_counter.update
            )
            _, update_result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )
            self._tb_log_update(
                explore_steps=int(self.step_counter.exploration),
                update_result=update_result,
            )

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
    ) -> Tuple[float, float]:
        self.heatup(heatup_steps)
        next_eval_step_limt = (
            self.agent.step_counter.exploration + explore_steps_between_eval
        )
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )
            quality, reward = self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        return quality, reward

    def _tb_add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._tb_writer is None:
            return
        try:
            self._tb_writer.add_scalar(tag, value, step)
        except Exception as exc:  # noqa: BLE001
            # Never crash training due to UI logging.
            self.logger.debug("TensorBoard add_scalar failed (%s): %s", tag, exc)

    def _tb_log_eval(
        self,
        *,
        explore_steps: int,
        reward: float,
        quality: float,
        info_results: dict,
    ) -> None:
        if self._tb_writer is None:
            return
        self._tb_add_scalar("eval/reward", float(reward), explore_steps)
        self._tb_add_scalar("eval/quality", float(quality), explore_steps)
        self._tb_add_scalar(
            "eval/best_quality", float(self.best_eval["quality"]), explore_steps
        )
        for name, value in info_results.items():
            self._tb_add_scalar(f"eval/{name}", float(value), explore_steps)
        self._tb_add_scalar("steps/exploration", float(explore_steps), explore_steps)
        self._tb_add_scalar(
            "steps/update", float(self.step_counter.update), explore_steps
        )
        self._tb_add_scalar(
            "episodes/exploration", float(self.episode_counter.exploration), explore_steps
        )

    def _tb_log_update(self, *, explore_steps: int, update_result: Any) -> None:
        if self._tb_writer is None:
            return
        losses = self._extract_loss_triplets(update_result)
        if not losses:
            return
        q1 = sum(v[0] for v in losses) / len(losses)
        q2 = sum(v[1] for v in losses) / len(losses)
        policy = sum(v[2] for v in losses) / len(losses)
        self._tb_add_scalar("loss/q1", float(q1), explore_steps)
        self._tb_add_scalar("loss/q2", float(q2), explore_steps)
        self._tb_add_scalar("loss/policy", float(policy), explore_steps)
        self._tb_add_scalar("loss/update_steps", float(len(losses)), explore_steps)

    @staticmethod
    def _extract_loss_triplets(update_result: Any) -> List[Tuple[float, float, float]]:
        """Normalize various update() return shapes to [(q1, q2, policy), ...]."""
        if update_result is None:
            return []
        if not isinstance(update_result, list):
            return []
        if not update_result:
            return []

        first = update_result[0]

        # Common case: [[q1, q2, policy], ...]
        if isinstance(first, (list, tuple)):
            triplets: List[Tuple[float, float, float]] = []
            for item in update_result:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                try:
                    triplets.append((float(item[0]), float(item[1]), float(item[2])))
                except (TypeError, ValueError):
                    continue
            return triplets

        # Fallback: a single flat list [q1, q2, policy]
        if isinstance(first, (int, float)) and len(update_result) >= 3:
            try:
                return [
                    (
                        float(update_result[0]),
                        float(update_result[1]),
                        float(update_result[2]),
                    )
                ]
            except (TypeError, ValueError):
                return []
        return []
