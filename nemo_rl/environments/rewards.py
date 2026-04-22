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
import re
from typing import Callable, Optional

import numpy as np
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

# initialize math_verify_func once
math_verify_func = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(),
    ),
)

boxed = lambda x: "\\boxed{" + x + "}" if not x.startswith("\\boxed{") else x


def math_expression_reward(
    ground_truth: str, response: str, tag: str = "answer"
) -> tuple[float, bool]:
    """Reward the agent when the answer within the <{tag}> tags is the same expression as the ground truth.

    The `tag` is customizable and must be specified as part of the user COT prompt text file.
    """
    match = re.search(rf"<{tag}>([\s\S]*)</{tag}>", response)
    if match:
        answer = match.group(1)
        try:
            score, _ = math_verify_func([boxed(ground_truth)], [boxed(answer)])
            return float(score), score > 0.1
        except (Exception, TimeoutException) as e:
            return 0.0, False
    return 0.0, False


def format_reward(
    ground_truth: str,
    response: str,
    think_tag: str = "think",
    answer_tag: str = "answer",
) -> tuple[float, Optional[bool]]:
    """Reward the agent when the response follows the format: (.*) <think> (.*) </think> <answer> (.*) </answer>.

    The `think_tag` and `answer_tag` are customizable and must be specified as part of the user COT prompt text file.
    """
    rew = 0.0
    if re.search(rf"<{think_tag}>[\s\S]*</{think_tag}>", response):
        rew += 0.25  # 0.25 points for having think tags
    if re.search(rf"<{answer_tag}>[\s\S]*</{answer_tag}>", response):
        rew += 0.75  # 0.75 points for having answer tags
    return rew, None


def exact_answer_alphanumeric_reward(
    ground_truth: str, response: str, answer_tag: str = "answer"
) -> tuple[float, bool]:
    """Reward the agent when the answer within the <{answer_tag}> tags is the same as the ground truth (case-insensitive).

    The `answer_tag` is customizable and must be specified as part of the user COT prompt text file.
    """
    match = re.search(rf"<{answer_tag}>([\s\S]*)</{answer_tag}>", response)
    if match:
        answer = match.group(1)
        # Remove all non-alphanumeric characters (including whitespace, punctuation, etc.)
        answer_clean = "".join(c for c in answer if c.isalnum()).lower()
        ground_truth_clean = "".join(c for c in ground_truth if c.isalnum()).lower()
        if answer_clean == ground_truth_clean:
            return 1.0, True
    return 0.0, False


def bbox_giou_reward(
    ground_truth: str,
    response: str,
    giou_penalty_thres: float = 10.0,
    answer_tag: str = "answer",
) -> tuple[float, bool]:
    """Given [x1, y1, x2, y2] normalized bounding box coordinates within the <{answer_tag}> tags, compute the GIoU between the ground truth and the response.

    The `answer_tag` is customizable and must be specified as part of the user COT prompt text file.
    """
    match = re.search(rf"<{answer_tag}>([\s\S]*)</{answer_tag}>", response)
    if match:
        answer = match.group(1)
    else:
        return 0.0, False

    try:
        x1g, y1g, x2g, y2g = [
            float(x) for x in ground_truth.replace("[", "").replace("]", "").split(",")
        ]
        x1r, y1r, x2r, y2r = [
            float(x) for x in answer.replace("[", "").replace("]", "").split(",")
        ]
    except ValueError:
        return 0.0, False

    # compute iou function
    # compute the area of the ground truth and response bounding boxes
    area_g = (x2g - x1g) * (y2g - y1g)
    area_r = (x2r - x1r) * (y2r - y1r)
    # compute the intersection of the ground truth and response bounding boxes
    x1i = max(x1g, x1r)
    y1i = max(y1g, y1r)
    x2i = min(x2g, x2r)
    y2i = min(y2g, y2r)
    # compute the area of the intersection
    area_i = max(0.0, x2i - x1i) * max(0.0, y2i - y1i)
    # compute the area of the union
    area_u = max(1e-3, area_g + area_r - area_i)
    # compute the iou
    iou = area_i / area_u
    # if iou is too low, introduce a giou term to compensate
    if iou < giou_penalty_thres:
        # compute convex hull as min
        x1c = min(x1g, x1r)
        y1c = min(y1g, y1r)
        x2c = max(x2g, x2r)
        y2c = max(y2g, y2r)
        # compute the area of the convex hull
        area_c = max(1e-3, (x2c - x1c) * (y2c - y1c))
        # compute the giou
        giou = iou - (area_c - area_u) / area_c
    else:
        giou = iou
    return giou, giou > 0.5


def svg_format_reward(ground_truth: str, response: str) -> tuple[float, Optional[bool]]:
    """Per-sample SVG format gate: 1.0 iff response contains a renderable <svg>…</svg>.

    ground_truth is ignored (may be a Sprint-6 JSON envelope; parsing is done by batch
    rewards). Returns (score, None) — None means "don't contribute to is_correct".
    """
    from nemo_rl.hackaton.rewards.r_fmt import r_fmt

    return r_fmt(response).score, None


# ===== Group-aware batch rewards (judge + solve_aux) =====
# Unlike per-sample rewards above, these take (gts_list, responses_list) at once and
# return list[float] in the SAME ORDER as the input. They group by ground_truth (same
# gt = same GRPO group, judged jointly by a VL model).

_BATCH_REWARD_NAMES = frozenset({"svg_judge_group", "svg_solve_aux"})


def _parse_gt_payload(gt: str) -> Optional[dict]:
    """Parse the JSON ground-truth envelope. Returns None if gt is not the v1 envelope."""
    import json

    try:
        d = json.loads(gt)
    except Exception:
        return None
    if not isinstance(d, dict) or d.get("_format") != "sprint6_v1":
        return None
    for k in ("question", "image_b64", "answer"):
        if k not in d:
            return None
    return d


def _group_by_gt(
    gts: list[str], responses: list[str]
) -> tuple[dict[str, list[int]], list[Optional[dict]]]:
    """Group indices by ground_truth string.

    Returns (groups_map, parsed_payloads):
        groups_map[gt_string]   = [list of sample indices sharing that gt].
        parsed_payloads[i]      = parsed JSON dict (or None if not the v1 envelope).
    Samples whose gt does not parse as the envelope are skipped (stay at 0.0).
    """
    groups: dict[str, list[int]] = {}
    parsed: list[Optional[dict]] = []
    for i, gt in enumerate(gts):
        pd = _parse_gt_payload(gt)
        parsed.append(pd)
        if pd is not None:
            groups.setdefault(gt, []).append(i)
    return groups, parsed


def _run_group_calls(
    gts: list[str], responses: list[str], group_call_async: Callable
) -> list[float]:
    """Group by gt, run one async call per group inside a single asyncio.gather, scatter back.

    group_call_async signature:
        async (client, question, problem_image_png_bytes, responses, [gold_answer]) -> list[float]
    ``gold_answer`` is passed iff the callable accepts it (solver accepts, judge doesn't);
    detected via inspect.signature.
    """
    import asyncio
    import base64
    import inspect

    import httpx

    out = [0.0] * len(responses)
    groups, parsed = _group_by_gt(gts, responses)
    if not groups:
        return out

    takes_gold = "gold_answer" in inspect.signature(group_call_async).parameters

    async def _runner():
        async with httpx.AsyncClient(
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=64)
        ) as client:
            group_tasks = []
            group_idx_list = []
            for gt_str, idx_list in groups.items():
                pd = parsed[idx_list[0]]
                assert pd is not None
                img_bytes = base64.b64decode(pd["image_b64"])
                group_responses = [responses[i] for i in idx_list]
                kwargs = dict(
                    client=client,
                    question=pd["question"],
                    problem_image_png_bytes=img_bytes,
                    responses=group_responses,
                )
                if takes_gold:
                    kwargs["gold_answer"] = pd["answer"]
                group_tasks.append(group_call_async(**kwargs))
                group_idx_list.append(idx_list)
            # Fire ALL groups concurrently.
            group_scores_list = await asyncio.gather(*group_tasks, return_exceptions=True)
            for idx_list, g_scores in zip(group_idx_list, group_scores_list):
                if isinstance(g_scores, Exception):
                    # Graceful degrade: leave as 0.0.
                    print(f"[batch reward] group call failed: {g_scores!r}")
                    continue
                for local_i, global_i in enumerate(idx_list):
                    out[global_i] = float(g_scores[local_i])

    asyncio.run(_runner())
    return out


def svg_judge_group_scores(
    ground_truths: list[str], responses: list[str]
) -> list[float]:
    """Batch-level reward. Returns judge scores in [0,1] per response."""
    try:
        from nemo_rl.hackaton.rewards.r_judge import r_judge_group

        return _run_group_calls(ground_truths, responses, r_judge_group)
    except Exception as e:
        print(f"[svg_judge_group_scores] fatal error → [0]*n: {e!r}")
        return [0.0] * len(responses)


def svg_solve_aux_scores(
    ground_truths: list[str], responses: list[str]
) -> list[float]:
    """Batch-level reward. Returns scheme-B {0,1,2} per response."""
    try:
        from nemo_rl.hackaton.rewards.r_solve import r_solve_group

        return _run_group_calls(ground_truths, responses, r_solve_group)
    except Exception as e:
        print(f"[svg_solve_aux_scores] fatal error → [0]*n: {e!r}")
        return [0.0] * len(responses)


def combine_reward_functions(
    reward_functions: list[tuple[Callable[[str, str], tuple[float, bool]], float]],
) -> Callable[[str, str], tuple[float, bool]]:
    """Returns a callable function that takes (ground_truth, response) and collects multiple reward functions in sequence.

    The reward functions are weighted by the second element of the tuple.
    This information can be provided in the YAML config file and resolved in the VLMEnvironment class.

    Args:
        reward_functions: list[tuple[Callable[[str, str], tuple[float, bool]], float]]. A list of reward functions and their weights.

    Returns:
        Callable[[str, str], tuple[float, bool]]: A callable function that takes (ground_truth, response) and collects multiple reward functions in sequence
    """
    weights = [weight for _, weight in reward_functions]
    weights = np.array(weights) / np.sum(weights)  # renormalize weights to 1

    def combined_reward_func(ground_truth: str, response: str) -> tuple[float, bool]:
        reward_env_score = [
            reward_func(ground_truth, response) for reward_func, _ in reward_functions
        ]
        rewards = [x[0] for x in reward_env_score]
        is_correct = [
            x[1] for x in reward_env_score if x[1] is not None
        ]  # skip None values, because they do not contribute to the "correctness" of the response (e.g. format_reward, because the answer can still be correct without <think> tags)
        is_correct = all(is_correct)
        return np.sum(np.array(rewards) * weights), is_correct

    return combined_reward_func
