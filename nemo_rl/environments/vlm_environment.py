# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import io
import logging
from functools import partial
from typing import Any, Callable, List, Optional, TypedDict

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.rewards import (
    _BATCH_REWARD_NAMES,
    bbox_giou_reward,
    combine_reward_functions,
    exact_answer_alphanumeric_reward,
    format_reward,
    math_expression_reward,
    svg_format_reward,
    svg_judge_group_scores,
    svg_solve_aux_scores,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class VLMEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]  # Default stop strings for this env
    reward_functions: List[dict[str, Any]]  # list of reward functions and their weights


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


@ray.remote
class VLMVerifyWorker:
    def __init__(self, cfg: VLMEnvConfig) -> None:
        logging.getLogger("vlm_worker").setLevel(logging.CRITICAL)
        # Split reward functions into two kinds:
        #   - per-sample: (gt, resp) -> (score, is_correct)
        #   - batch-level: (gts_list, resps_list) -> list[float]
        # Batch rewards need the full group to compute (judge / pairwise solve).
        reward_functions: list[tuple[Callable, float]] = []
        batch_rewards: list[
            tuple[str, Callable[[list[str], list[str]], list[float]], float]
        ] = []

        for reward_func_cfg in cfg["reward_functions"]:
            reward_func_name: str = reward_func_cfg["name"]
            reward_func_weight: float = reward_func_cfg["weight"]
            reward_func_kwargs: Optional[dict] = reward_func_cfg.get("kwargs", None)

            if reward_func_name in _BATCH_REWARD_NAMES:
                if reward_func_name == "svg_judge_group":
                    batch_fn: Callable[[list[str], list[str]], list[float]] = (
                        svg_judge_group_scores
                    )
                elif reward_func_name == "svg_solve_aux":
                    batch_fn = svg_solve_aux_scores
                else:
                    raise ValueError(f"Unknown batch reward: {reward_func_name}")
                if reward_func_kwargs is not None:
                    batch_fn = partial(batch_fn, **reward_func_kwargs)
                batch_rewards.append((reward_func_name, batch_fn, reward_func_weight))
                continue

            reward_func: Callable[[str, str], tuple[float, Optional[bool]]]
            if reward_func_name == "format":
                reward_func = format_reward
            elif reward_func_name == "exact_alnum":
                reward_func = exact_answer_alphanumeric_reward
            elif reward_func_name == "math_expr":
                reward_func = math_expression_reward
            elif reward_func_name == "bbox_giou":
                reward_func = bbox_giou_reward
            elif reward_func_name == "svg_format":
                reward_func = svg_format_reward
            else:
                raise ValueError(f"Invalid reward function: {reward_func_name}")

            if reward_func_kwargs is not None:
                reward_func = partial(reward_func, **reward_func_kwargs)

            reward_functions.append((reward_func, reward_func_weight))

        if len(reward_functions) == 0 and len(batch_rewards) == 0:
            raise ValueError("No reward functions provided")

        total_weight = sum(w for _, w in reward_functions) + sum(
            w for _, _, w in batch_rewards
        )
        if total_weight <= 0:
            raise ValueError("Reward function weights sum to zero")

        self.per_sample_pairs = reward_functions
        self.batch_pairs = batch_rewards
        self._weight_norm = total_weight
        # Kept for backward compat — only used in the no-batch case below.
        self.verify_func = (
            combine_reward_functions(reward_functions) if reward_functions else None
        )

    def verify(
        self, pred_responses: list[str], ground_truths: list[str]
    ) -> list[float]:
        """Compute rewards for a batch of (response, ground_truth) pairs.

        Per-sample rewards (e.g. svg_format) are computed first; then batch-level
        rewards (svg_judge_group, svg_solve_aux) are called once over the full
        batch. Final per-response reward is the weighted sum of both, renormalized
        by the total configured weight.
        """
        n = len(pred_responses)

        per_sample_accum = [0.0] * n
        for reward_func, w in self.per_sample_pairs:
            for i, (response, gt) in enumerate(zip(pred_responses, ground_truths)):
                try:
                    with _mute_output():
                        s, _ = reward_func(gt, response)
                    per_sample_accum[i] += float(s) * w
                except Exception as e:
                    print(f"Error in per-sample reward {reward_func}: {e!r}")

        batch_accum = [0.0] * n
        for name, batch_fn, w in self.batch_pairs:
            try:
                scores = batch_fn(ground_truths, pred_responses)
            except Exception as e:
                print(f"Error in batch reward {name}: {e!r}")
                scores = [0.0] * n
            if len(scores) != n:
                print(
                    f"Batch reward {name} returned {len(scores)} scores for {n} "
                    f"responses — zeroing"
                )
                scores = [0.0] * n
            for i, s in enumerate(scores):
                batch_accum[i] += float(s) * w

        return [
            (per_sample_accum[i] + batch_accum[i]) / self._weight_norm for i in range(n)
        ]


class VLMEnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote(max_restarts=-1, max_task_retries=-1)
class VLMEnvironment(EnvironmentInterface):
    def __init__(self, cfg: VLMEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            VLMVerifyWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote(cfg)
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(  # type: ignore[override]
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[VLMEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the vlm environment.

        Args:
            message_log: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the VLM.
            metadata: list[VLMEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - list[dict[str, str]]: Observations/responses batch
                - list[dict]: Updated metadata
                - list[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # Extract the assistant's responses from the message history
        # Each message list should have at least one assistant response
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(chunk, ground_truth_chunk)
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]

        results = ray.get(futures)

        # flatten the results
        results = [item for sublist in results for item in sublist]
        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if result
                else "Environment: incorrect",
            }
            for result in results
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        """
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics = {
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
