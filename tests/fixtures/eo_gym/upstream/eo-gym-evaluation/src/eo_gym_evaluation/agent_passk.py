#!/usr/bin/env python3
"""
Evaluate Agent_TEST_outputs JSONL files with early-stop Pass@k for open-text QA.

Highlights:
- Semantic final-answer matching via gpt-4.1-mini (cache-backed).
- Counting-question focus analysis:
    - whether the right tools were called
    - whether tool input arguments are semantically equivalent to GT arguments
- Early-stop trajectory evaluation (traj0 -> traj1 -> traj2 until first correct).
- GT tool-step evaluation using only GT `messages` as expected-tool source.
- Split outputs by ground:
    - all-tools
    - mapped-build-in-functions
- Per-ground CSVs + quick-check charts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

from eo_gym_data_synthesis.prompt_manager import (
    SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT,
    TOOL_ARGUMENT_SEMANTIC_JUDGE_SYSTEM_PROMPT,
)
from eo_gym_evaluation.question_taxonomy import (
    build_question_type_bucket_map,
    question_type_bucket_display_name,
    normalize_question_type_key as shared_normalize_question_type_key,
    question_type_bucket_sort_key,
    question_type_to_bucket as shared_question_type_to_bucket,
    resolve_question_type_and_bucket,
)

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional at runtime
    OpenAI = None  # type: ignore

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional at runtime
    tqdm = None  # type: ignore


FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*", re.IGNORECASE)
TRAJ_SUFFIX_RE = re.compile(r":traj(\d+)$")
JSON_START_RE = re.compile(r"\s*[\[{]")
PROMPT_VERSION_SUFFIX_RE = re.compile(r"([_-])(simple|complex)[_-]prompt$", re.IGNORECASE)

PROMPT_VERSION_PRIORITY: Dict[str, int] = {
    "simple-prompt": 0,
    "complex-prompt": 1,
    "unknown": 2,
}

GROUND_BY_MODE: Dict[Tuple[str, str], str] = {
    ("all_tools", "builtin_function_call"): "all-tools",
    ("dataset_mapped", "builtin_function_call"): "mapped-build-in-functions",
}

JUDGE_PROMPT_VERSION = "semantic_match_v1_open_text"
PARAMETER_JUDGE_PROMPT_VERSION = "parameter_semantic_match_v1_counting"

SUMMARY_FIELDNAMES = [
    "ground",
    "model",
    "file",
    "prompt_version",
    "training_stage",
    "question_total",
    "accuracy",
    "pass@1",
    "pass@2",
    "pass@3",
    "avg_tries_used",
    "reference_steps_total",
    "tool_calls_consumed_total",
    "efficiency_consumed",
    "avg_tool_calls_consumed_per_question",
    "illegal_function_call_error_rate_consumed",
    "tools_any_order_rate_consumed",
    "tools_in_order_rate_consumed",
    "tools_exact_match_rate_consumed",
    "parameters_score_mean_consumed",
    "semantic_judged_trajectories",
    "semantic_cache_hits",
    "semantic_api_calls",
    "no_function_call_trajectories_consumed",
    "no_function_call_rate_consumed",
    "missing_final_answer_marker_consumed",
    "missing_gt_questions",
]

QUESTION_OUTCOME_FIELDNAMES = [
    "ground",
    "model",
    "file",
    "question_id",
    "question_type",
    "question_type_bucket",
    "question",
    "gt_answer",
    "reference_steps",
    "available_traj_ids",
    "consumed_traj_ids",
    "tries_used",
    "first_correct_traj",
    "accuracy",
    "pass@1",
    "pass@2",
    "pass@3",
    "consumed_tool_calls",
    "consumed_function_call_errors",
    "consumed_missing_final_answer_markers",
]

TRAJECTORY_METRICS_FIELDNAMES = [
    "ground",
    "model",
    "file",
    "question_id",
    "question_type",
    "question_type_bucket",
    "traj_id",
    "consumed",
    "judged",
    "is_correct",
    "semantic_reason",
    "semantic_cache_hit",
    "semantic_api_called",
    "function_calls",
    "function_call_errors",
    "function_call_error_rate",
    "final_answer",
    "final_answer_missing_marker",
    "tools_any_order",
    "tools_in_order",
    "tools_exact_match",
    "parameters_score",
    "expected_steps",
    "actual_steps",
    "parameter_matched_expected_steps",
]

STEP_DETAILS_FIELDNAMES = [
    "ground",
    "model",
    "file",
    "question_id",
    "question_type",
    "question_type_bucket",
    "traj_id",
    "consumed",
    "judged",
    "is_correct",
    "expected_step_index",
    "expected_tool",
    "actual_step_index",
    "actual_tool",
    "tool_name_match",
    "args_match",
    "output_match",
    "step_match",
    "expected_arguments_norm",
    "actual_arguments_norm",
    "expected_output_norm",
    "actual_output_norm",
]

TRAJ_LENGTH_FIELDNAMES = [
    "ground",
    "model",
    "file",
    "traj_id",
    "function_calls",
    "count",
    "proportion",
]

TRIES_USED_FIELDNAMES = [
    "ground",
    "model",
    "file",
    "tries_used",
    "count",
    "solved_count",
    "unsolved_count",
    "proportion",
]

FUNCTION_CALL_ERROR_FIELDNAMES = [
    "ground",
    "model",
    "file",
    "function_calls",
    "function_call_errors",
    "function_call_error_rate",
]

TOOL_STEP_SUMMARY_FIELDNAMES = [
    "ground",
    "model",
    "file",
    "trajectory_count",
    "tools_any_order_rate",
    "tools_in_order_rate",
    "tools_exact_match_rate",
    "parameters_score_mean",
]

QUESTION_TYPE_BREAKDOWN_FIELDNAMES = [
    "ground",
    "model",
    "file",
    "question_type_bucket",
    "question_total",
    "accuracy",
    "pass@1",
    "pass@2",
    "pass@3",
    "avg_tries_used",
    "reference_steps_total",
    "tool_calls_consumed_total",
    "efficiency_consumed",
    "illegal_function_call_error_rate_consumed",
    "tools_any_order_rate_consumed",
    "tools_in_order_rate_consumed",
    "tools_exact_match_rate_consumed",
    "parameters_score_mean_consumed",
]

NO_FUNCTION_CALL_FIELDNAMES = [
    "ground",
    "model",
    "file",
    "trajectory_total",
    "no_function_call_trajectories",
    "no_function_call_rate",
]

COUNTING_FOCUS_SUMMARY_FIELDNAMES = [
    "ground",
    "model",
    "file",
    "counting_question_total",
    "counting_trajectory_total",
    "right_tool_any_order_rate",
    "right_tool_in_order_rate",
    "right_tool_exact_match_rate",
    "tool_name_matched_steps",
    "args_exact_match_rate",
    "args_semantic_same_rate",
    "args_semantic_same_steps",
    "args_semantic_cache_hits",
    "args_semantic_api_calls",
]

COUNTING_FOCUS_STEP_FIELDNAMES = [
    "ground",
    "model",
    "file",
    "question_id",
    "question_type",
    "question_type_bucket",
    "traj_id",
    "consumed",
    "expected_step_index",
    "expected_tool",
    "actual_step_index",
    "actual_tool",
    "args_match",
    "args_semantic_same",
    "args_semantic_reason",
    "args_semantic_cache_hit",
    "args_semantic_api_called",
    "expected_arguments_norm",
    "actual_arguments_norm",
]

TOOL_RENAME_COMPARISON_FIELDNAMES = [
    "ground",
    "prompt_version",
    "prompt_label",
    "variant",
    "comparison_label",
    "comparison_order",
    "stage",
    "is_tool_rename",
    "model",
    "file",
    "question_total",
    "pass@1",
    "pass@2",
    "pass@3",
    "accuracy",
    "efficiency_consumed",
    "no_function_call_trajectories_consumed",
    "no_function_call_rate_consumed",
    "delta_pass@1_vs_stage_baseline",
    "delta_pass@2_vs_stage_baseline",
    "delta_pass@3_vs_stage_baseline",
    "delta_accuracy_vs_stage_baseline",
]

PROMPT_VERSION_COMPARISON_FIELDNAMES = [
    "ground",
    "comparison_group",
    "comparison_group_label",
    "model",
    "training_stage",
    "simple_prompt_files",
    "complex_prompt_files",
    "question_total_simple",
    "question_total_complex",
    "pass@1_simple",
    "pass@1_complex",
    "delta_pass@1_complex_minus_simple",
    "pass@2_simple",
    "pass@2_complex",
    "delta_pass@2_complex_minus_simple",
    "pass@3_simple",
    "pass@3_complex",
    "delta_pass@3_complex_minus_simple",
    "accuracy_simple",
    "accuracy_complex",
    "delta_accuracy_complex_minus_simple",
]

OVERALL_MODEL_METRICS_PROMPT_SPLIT_METRICS = [
    "question_total",
    "trajectory_total_consumed",
    "accuracy",
    "pass@1",
    "pass@2",
    "pass@3",
    "avg_tries_used",
    "reference_steps_total",
    "tool_calls_consumed_total",
    "efficiency_consumed",
    "avg_tool_calls_consumed_per_question",
    "illegal_function_call_errors_consumed",
    "illegal_function_call_error_rate_consumed",
    "tools_any_order_rate_consumed",
    "tools_in_order_rate_consumed",
    "tools_exact_match_rate_consumed",
    "parameters_score_mean_consumed",
    "semantic_judged_trajectories",
    "semantic_cache_hits",
    "semantic_api_calls",
    "no_function_call_trajectories_consumed",
    "no_function_call_rate_consumed",
    "missing_final_answer_marker_consumed",
    "missing_gt_questions",
    "consumed_function_calls_mean",
    "consumed_function_calls_median",
    "consumed_function_calls_max",
    "counting_question_total",
    "counting_trajectory_total",
    "counting_right_tool_any_order_rate",
    "counting_right_tool_in_order_rate",
    "counting_right_tool_exact_match_rate",
    "counting_tool_name_matched_steps",
    "counting_args_exact_match_rate",
    "counting_args_semantic_same_rate",
    "counting_args_semantic_same_steps",
    "counting_args_semantic_cache_hits",
    "counting_args_semantic_api_calls",
]

OVERALL_MODEL_METRICS_PROMPT_SPLIT_DELTA_FIELDS = [
    "delta_accuracy_complex_minus_simple",
    "delta_pass@1_complex_minus_simple",
    "delta_pass@2_complex_minus_simple",
    "delta_pass@3_complex_minus_simple",
    "delta_avg_tries_used_complex_minus_simple",
    "delta_efficiency_consumed_complex_minus_simple",
    "delta_avg_tool_calls_consumed_per_question_complex_minus_simple",
    "delta_illegal_function_call_error_rate_consumed_complex_minus_simple",
    "delta_no_function_call_rate_consumed_complex_minus_simple",
    "delta_consumed_function_calls_mean_complex_minus_simple",
    "delta_counting_args_semantic_same_rate_complex_minus_simple",
]

OVERALL_MODEL_METRICS_PROMPT_SPLIT_FIELDNAMES = [
    "ground",
    "comparison_group",
    "comparison_group_label",
    "model",
    "training_stage",
    "has_simple_prompt",
    "has_complex_prompt",
    "simple_prompt_files",
    "complex_prompt_files",
    "simple_prompt_file_count",
    "complex_prompt_file_count",
] + [
    f"{metric}_simple" for metric in OVERALL_MODEL_METRICS_PROMPT_SPLIT_METRICS
] + [
    f"{metric}_complex" for metric in OVERALL_MODEL_METRICS_PROMPT_SPLIT_METRICS
] + OVERALL_MODEL_METRICS_PROMPT_SPLIT_DELTA_FIELDS

QUESTION_TYPE_BUCKET_PLOT_FIELDNAMES = [
    "prompt_version",
    "question_type_bucket",
    "model",
    "question_total",
    "pass@1",
    "pass@2",
    "pass@3",
]
QUESTION_TYPE_BUCKET_MAP = build_question_type_bucket_map()
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IGNORED_QUESTION_TYPE_KEYS: Set[str] = {
    #"cross-modal analysis",
}


@dataclass
class ExpectedStep:
    step_index: int
    tool_call_id: str
    function_name: str
    arguments_raw: Any
    output_raw: Any
    arguments_norm: Any
    output_norm: Any


@dataclass
class GroundTruthItem:
    question_id: str
    question: str
    gt_answer: str
    expected_steps: List[ExpectedStep]


@dataclass
class TrajectoryStep:
    step_index: int
    function_name: str
    arguments_raw: Any
    observation_raw: Any
    arguments_norm: Any
    observation_norm: Any
    is_error: bool
    error_type: str


@dataclass
class TrajectoryRecord:
    ground: str
    model: str
    file: str
    question_id: str
    question: str
    traj_id: int
    uid: str
    steps: List[TrajectoryStep]
    final_answer: str
    final_answer_missing_marker: bool

    @property
    def function_calls(self) -> int:
        return len(self.steps)

    @property
    def function_call_errors(self) -> int:
        return sum(1 for step in self.steps if step.is_error)


@dataclass
class FileEvaluation:
    ground: str
    model: str
    file: str
    prompt_version: str
    question_rows: List[Dict[str, Any]]
    trajectory_rows: List[Dict[str, Any]]
    step_rows: List[Dict[str, Any]]
    missing_gt_questions: int


@dataclass
class QuestionEvaluation:
    question_id: str
    missing_gt: bool
    question_row: Optional[Dict[str, Any]]
    trajectory_rows: List[Dict[str, Any]]
    step_rows: List[Dict[str, Any]]


class SemanticMatcher:
    def __init__(
        self,
        *,
        cache_path: Path,
        judge_model: str,
        prompt_version: str,
        max_retries: int = 5,
        retry_sleep_sec: float = 2.0,
    ) -> None:
        self.cache_path = cache_path
        self.judge_model = judge_model
        self.prompt_version = prompt_version
        self.max_retries = max_retries
        self.retry_sleep_sec = retry_sleep_sec

        self.cache: Dict[str, Dict[str, Any]] = {}
        self._cache_append_fh = None
        self._client_local = threading.local()
        self._openai_available = OpenAI is not None
        self._cache_lock = threading.Lock()
        self._inflight: Dict[str, threading.Event] = {}
        self._load_cache()

    def close(self) -> None:
        if self._cache_append_fh is not None:
            try:
                self._cache_append_fh.close()
            except Exception:
                pass
            self._cache_append_fh = None

    def _load_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    continue
                key_hash = str(obj.get("key_hash") or "").strip()
                if not key_hash:
                    continue
                self.cache[key_hash] = obj

    def _get_client(self):
        cached = getattr(self._client_local, "client", None)
        if cached is not None:
            return cached
        if not self._openai_available:
            return None
        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            client = OpenAI()
            self._client_local.client = client
            return client
        except Exception:
            return None

    def _append_cache_record(self, record: Dict[str, Any]) -> None:
        if self._cache_append_fh is None:
            self._cache_append_fh = self.cache_path.open("a", encoding="utf-8")
        self._cache_append_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cache_append_fh.flush()

    def _cache_key(
        self,
        *,
        question: str,
        gt_answer: str,
        pred_answer: str,
    ) -> str:
        payload = {
            "judge_model": self.judge_model,
            "prompt_version": self.prompt_version,
            "question": _normalize_text_for_cache(question),
            "gt_answer": _normalize_text_for_cache(gt_answer),
            "pred_answer": _normalize_text_for_cache(pred_answer),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def judge(
        self,
        *,
        question: str,
        gt_answer: str,
        pred_answer: str,
    ) -> Tuple[bool, str, bool, bool]:
        key_hash = self._cache_key(question=question, gt_answer=gt_answer, pred_answer=pred_answer)
        with self._cache_lock:
            cached = self.cache.get(key_hash)
            if cached is not None:
                return bool(cached.get("is_same_meaning", False)), str(cached.get("reason", "")), True, False

            inflight_event = self._inflight.get(key_hash)
            if inflight_event is None:
                inflight_event = threading.Event()
                self._inflight[key_hash] = inflight_event
                owner = True
            else:
                owner = False

        if not owner:
            inflight_event.wait()
            with self._cache_lock:
                cached = self.cache.get(key_hash)
            if cached is not None:
                return bool(cached.get("is_same_meaning", False)), str(cached.get("reason", "")), True, False
            fallback = _normalize_text_for_cache(gt_answer) == _normalize_text_for_cache(pred_answer)
            return fallback, "inflight_wait_missing_cache_fallback_exact", False, False

        result = False
        reason = ""
        try:
            result, reason = self._judge_via_api(question=question, gt_answer=gt_answer, pred_answer=pred_answer)
            now = datetime.now(timezone.utc).isoformat()
            record = {
                "key_hash": key_hash,
                "judge_model": self.judge_model,
                "prompt_version": self.prompt_version,
                "is_same_meaning": bool(result),
                "reason": reason,
                "created_at": now,
            }
            with self._cache_lock:
                self.cache[key_hash] = record
                self._append_cache_record(record)
            return bool(result), reason, False, True
        finally:
            with self._cache_lock:
                event = self._inflight.pop(key_hash, None)
            if event is not None:
                event.set()

    def _judge_via_api(self, *, question: str, gt_answer: str, pred_answer: str) -> Tuple[bool, str]:
        # Fast path for empty answers to avoid unnecessary calls.
        if not str(pred_answer or "").strip():
            return False, "empty_prediction"
        if not str(gt_answer or "").strip():
            return False, "empty_ground_truth"

        client = self._get_client()
        if client is None:
            fallback = _normalize_text_for_cache(gt_answer) == _normalize_text_for_cache(pred_answer)
            return fallback, "openai_unavailable_fallback_exact"

        system_prompt = SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT
        payload = {
            "question": question,
            "reference_final_answer": gt_answer,
            "candidate_final_answer": pred_answer,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.judge_model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                )
                content = ""
                if resp.choices:
                    content = (resp.choices[0].message.content or "").strip()
                data = json.loads(content) if content else {}
                result = bool(data.get("is_same_meaning", False))
                reason = str(data.get("reason", "")).strip()
                if not reason:
                    reason = "no_reason"
                return result, reason
            except Exception as exc:
                if attempt >= self.max_retries:
                    fallback = _normalize_text_for_cache(gt_answer) == _normalize_text_for_cache(pred_answer)
                    reason = f"api_error_fallback_exact: {type(exc).__name__}"
                    return fallback, reason
                sleep_sec = self.retry_sleep_sec * attempt
                time.sleep(sleep_sec)

        fallback = _normalize_text_for_cache(gt_answer) == _normalize_text_for_cache(pred_answer)
        return fallback, "api_retry_exhausted_fallback_exact"


def _canonical_argument_text(value: Any) -> str:
    normalized = _normalize_value(value)
    try:
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(normalized)


class ParameterSemanticMatcher:
    """
    Semantic matcher for tool arguments.
    Uses cache + gpt-4.1-mini (or configured judge model) and falls back to exact
    normalized match when API is unavailable.
    """

    def __init__(
        self,
        *,
        cache_path: Path,
        judge_model: str,
        prompt_version: str,
        max_retries: int = 5,
        retry_sleep_sec: float = 2.0,
    ) -> None:
        self.cache_path = cache_path
        self.judge_model = judge_model
        self.prompt_version = prompt_version
        self.max_retries = max_retries
        self.retry_sleep_sec = retry_sleep_sec

        self.cache: Dict[str, Dict[str, Any]] = {}
        self._cache_append_fh = None
        self._client_local = threading.local()
        self._openai_available = OpenAI is not None
        self._cache_lock = threading.Lock()
        self._inflight: Dict[str, threading.Event] = {}
        self._load_cache()

    def close(self) -> None:
        if self._cache_append_fh is not None:
            try:
                self._cache_append_fh.close()
            except Exception:
                pass
            self._cache_append_fh = None

    def _load_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    continue
                key_hash = str(obj.get("key_hash") or "").strip()
                if not key_hash:
                    continue
                self.cache[key_hash] = obj

    def _get_client(self):
        cached = getattr(self._client_local, "client", None)
        if cached is not None:
            return cached
        if not self._openai_available:
            return None
        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            client = OpenAI()
            self._client_local.client = client
            return client
        except Exception:
            return None

    def _append_cache_record(self, record: Dict[str, Any]) -> None:
        if self._cache_append_fh is None:
            self._cache_append_fh = self.cache_path.open("a", encoding="utf-8")
        self._cache_append_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cache_append_fh.flush()

    def _cache_key(
        self,
        *,
        question: str,
        tool_name: str,
        reference_arguments: Any,
        candidate_arguments: Any,
    ) -> str:
        payload = {
            "judge_model": self.judge_model,
            "prompt_version": self.prompt_version,
            "question": _normalize_text_for_cache(question),
            "tool_name": _normalize_text_for_cache(tool_name),
            "reference_arguments": _canonical_argument_text(reference_arguments),
            "candidate_arguments": _canonical_argument_text(candidate_arguments),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def judge(
        self,
        *,
        question: str,
        tool_name: str,
        reference_arguments: Any,
        candidate_arguments: Any,
    ) -> Tuple[bool, str, bool, bool]:
        key_hash = self._cache_key(
            question=question,
            tool_name=tool_name,
            reference_arguments=reference_arguments,
            candidate_arguments=candidate_arguments,
        )
        with self._cache_lock:
            cached = self.cache.get(key_hash)
            if cached is not None:
                return bool(cached.get("is_same_meaning", False)), str(cached.get("reason", "")), True, False

            inflight_event = self._inflight.get(key_hash)
            if inflight_event is None:
                inflight_event = threading.Event()
                self._inflight[key_hash] = inflight_event
                owner = True
            else:
                owner = False

        if not owner:
            inflight_event.wait()
            with self._cache_lock:
                cached = self.cache.get(key_hash)
            if cached is not None:
                return bool(cached.get("is_same_meaning", False)), str(cached.get("reason", "")), True, False
            fallback = _canonical_argument_text(reference_arguments) == _canonical_argument_text(candidate_arguments)
            return fallback, "inflight_wait_missing_cache_fallback_exact", False, False

        result = False
        reason = ""
        try:
            result, reason = self._judge_via_api(
                question=question,
                tool_name=tool_name,
                reference_arguments=reference_arguments,
                candidate_arguments=candidate_arguments,
            )
            now = datetime.now(timezone.utc).isoformat()
            record = {
                "key_hash": key_hash,
                "judge_model": self.judge_model,
                "prompt_version": self.prompt_version,
                "is_same_meaning": bool(result),
                "reason": reason,
                "created_at": now,
            }
            with self._cache_lock:
                self.cache[key_hash] = record
                self._append_cache_record(record)
            return bool(result), reason, False, True
        finally:
            with self._cache_lock:
                event = self._inflight.pop(key_hash, None)
            if event is not None:
                event.set()

    def _judge_via_api(
        self,
        *,
        question: str,
        tool_name: str,
        reference_arguments: Any,
        candidate_arguments: Any,
    ) -> Tuple[bool, str]:
        ref_text = _canonical_argument_text(reference_arguments)
        cand_text = _canonical_argument_text(candidate_arguments)

        if ref_text == cand_text:
            return True, "exact_normalized_arguments_match"
        if not str(cand_text).strip():
            return False, "empty_candidate_arguments"
        if not str(ref_text).strip():
            return False, "empty_reference_arguments"

        client = self._get_client()
        if client is None:
            return False, "openai_unavailable_fallback_not_exact"

        system_prompt = TOOL_ARGUMENT_SEMANTIC_JUDGE_SYSTEM_PROMPT
        payload = {
            "question": question,
            "tool_name": tool_name,
            "reference_arguments": reference_arguments,
            "candidate_arguments": candidate_arguments,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.judge_model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                )
                content = ""
                if resp.choices:
                    content = (resp.choices[0].message.content or "").strip()
                data = json.loads(content) if content else {}
                result = bool(data.get("is_same_meaning", False))
                reason = str(data.get("reason", "")).strip()
                if not reason:
                    reason = "no_reason"
                return result, reason
            except Exception as exc:
                if attempt >= self.max_retries:
                    reason = f"api_error_fallback_not_exact: {type(exc).__name__}"
                    return False, reason
                sleep_sec = self.retry_sleep_sec * attempt
                time.sleep(sleep_sec)

        return False, "api_retry_exhausted_fallback_not_exact"


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            idx = 0
            while idx < len(line):
                try:
                    obj, idx = decoder.raw_decode(line, idx)
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict):
                    yield line_no, obj
                while idx < len(line) and line[idx].isspace():
                    idx += 1


def _normalize_text_for_cache(text: Any) -> str:
    return " ".join(str(text or "").strip().split()).lower()


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    mid = len(values_sorted) // 2
    if len(values_sorted) % 2 == 1:
        return values_sorted[mid]
    return (values_sorted[mid - 1] + values_sorted[mid]) / 2.0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_json_like(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return stripped
    if not JSON_START_RE.match(stripped):
        return stripped
    try:
        return json.loads(stripped)
    except Exception:
        return stripped


def _normalize_path_like(text: str) -> str:
    value = text.strip().replace("\\", "/")
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        value = parsed.path or ""
    markers = ["EO_GYM_DATA/", "image_cache/"]
    for marker in markers:
        idx = value.find(marker)
        if idx >= 0:
            value = value[idx:]
            break
    value = re.sub(r"/+", "/", value)
    value = re.sub(r"_([0-9a-f]{8,})(?=\.[a-zA-Z0-9]+$)", "_<id>", value)
    return value.lower().strip()


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    text = str(value).strip()
    if not text:
        return ""
    parsed = _parse_json_like(text)
    if parsed is not text:
        return _normalize_value(parsed)
    if "/" in text or "http://" in text or "https://" in text:
        return _normalize_path_like(text)
    compact = " ".join(text.split())
    return compact.lower()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, val in value.items():
            k = str(key).strip().lower()
            out[k] = _normalize_value(val)
        return {k: out[k] for k in sorted(out.keys())}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return _normalize_scalar(value)


def _values_match(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_values_match(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_values_match(a[k], b[k]) for k in a.keys())
    return a == b


def _value_snippet(value: Any, max_len: int = 500) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _extract_question_from_gt_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        lines = [ln.strip() for ln in content.splitlines() if ln.strip() and ln.strip() != "<image>"]
        text = " ".join(lines).strip()
        match = re.search(r"question\s*:\s*", text, flags=re.IGNORECASE)
        if match:
            text = text[match.end() :].strip()
        return text
    return ""


def _load_ground_truth(gt_jsonl: Path) -> Dict[str, GroundTruthItem]:
    gt_map: Dict[str, GroundTruthItem] = {}
    for _line_no, record in iter_jsonl(gt_jsonl):
        question_id = str(record.get("id") or "").strip()
        if not question_id:
            continue
        gt_answer = str(record.get("gt_answer") or "").strip()
        messages = record.get("messages")
        question = _extract_question_from_gt_messages(messages)
        expected_steps: List[ExpectedStep] = []

        tool_outputs: Dict[str, Any] = {}
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "tool":
                    continue
                tool_call_id = str(msg.get("tool_call_id") or "").strip()
                if not tool_call_id:
                    continue
                tool_outputs[tool_call_id] = msg.get("content")

            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "assistant":
                    continue
                tool_calls = msg.get("tool_calls")
                if not isinstance(tool_calls, list):
                    continue
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    call_id = str(tool_call.get("id") or f"step_{len(expected_steps)}")
                    fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                    function_name = str(fn.get("name") or "").strip()
                    arguments_raw = fn.get("arguments")
                    output_raw = tool_outputs.get(call_id)
                    expected_steps.append(
                        ExpectedStep(
                            step_index=len(expected_steps),
                            tool_call_id=call_id,
                            function_name=function_name,
                            arguments_raw=arguments_raw,
                            output_raw=output_raw,
                            arguments_norm=_normalize_value(arguments_raw),
                            output_norm=_normalize_value(output_raw),
                        )
                    )

        gt_map[question_id] = GroundTruthItem(
            question_id=question_id,
            question=question,
            gt_answer=gt_answer,
            expected_steps=expected_steps,
        )

    return gt_map


def _normalize_question_type_key(question_type: Any) -> str:
    return shared_normalize_question_type_key(question_type)


def _question_type_to_bucket(question_type: Any) -> str:
    return shared_question_type_to_bucket(
        question_type,
        question_type_bucket_map=QUESTION_TYPE_BUCKET_MAP,
        apply_xbd_alias=False,
        unknown_value="",
    )


def _should_ignore_question_type(question_type: Any) -> bool:
    return _normalize_question_type_key(question_type) in IGNORED_QUESTION_TYPE_KEYS


def _load_question_type_map(csv_path: Path) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return mapping
        if "uid" not in reader.fieldnames:
            raise ValueError(f"Question-type CSV missing required 'uid' column: {csv_path}")
        if "question_type" not in reader.fieldnames:
            raise ValueError(f"Question-type CSV missing required 'question_type' column: {csv_path}")

        for row in reader:
            uid = _uid_base(row.get("uid"))
            if not uid:
                continue
            question_type, question_type_bucket = resolve_question_type_and_bucket(
                row.get("question_type"),
                row.get("data_source"),
                question_type_bucket_map=QUESTION_TYPE_BUCKET_MAP,
                apply_xbd_alias=True,
                unknown_value="",
            )
            current = mapping.setdefault(uid, {"question_type": "", "question_type_bucket": "", "data_source": ""})
            if not current["question_type"] and question_type:
                current["question_type"] = question_type
            if not current["question_type_bucket"] and question_type_bucket:
                current["question_type_bucket"] = question_type_bucket
            if not current["data_source"]:
                current["data_source"] = str(row.get("data_source") or "").strip()
    return mapping


def _normalize_question_type(question_type: Any, data_source: Any) -> str:
    normalized_question_type, _ = resolve_question_type_and_bucket(
        question_type,
        data_source,
        question_type_bucket_map=QUESTION_TYPE_BUCKET_MAP,
        apply_xbd_alias=True,
        unknown_value="",
    )
    return normalized_question_type


def _question_type_bucket_from_row(row: Dict[str, Any]) -> str:
    bucket = str(row.get("question_type_bucket") or "").strip()
    if bucket:
        return bucket
    return shared_question_type_to_bucket(
        row.get("question_type"),
        data_source=row.get("data_source"),
        question_type_bucket_map=QUESTION_TYPE_BUCKET_MAP,
        apply_xbd_alias=True,
        unknown_value="unknown",
    )


def _infer_prompt_version(*values: Any) -> str:
    for value in values:
        unified = re.sub(r"[\s_]+", "-", str(value or "").strip().lower())
        if "simple-prompt" in unified:
            return "simple-prompt"
        if "complex-prompt" in unified:
            return "complex-prompt"
    return "unknown"


def _prompt_version_display(prompt_version: Any) -> str:
    value = str(prompt_version or "").strip().lower()
    if value == "simple-prompt":
        return "Simple Prompt"
    if value == "complex-prompt":
        return "Complex Prompt"
    return "Unknown Prompt"


def _strip_prompt_version_suffix(text: Any) -> str:
    return PROMPT_VERSION_SUFFIX_RE.sub("", str(text or "").strip())


def _infer_training_stage(*values: Any) -> str:
    combined = " ".join(str(value or "") for value in values).strip().lower()
    unified = re.sub(r"[^a-z0-9]+", "-", combined).strip("-")
    if re.search(r"(?:^|-)grpo(?:$|-)", unified) or re.search(r"(?:^|-)rl(?:$|-)", unified):
        return "rl"
    if re.search(r"(?:^|-)sft(?:$|-)", unified):
        return "sft"
    return "base"


def _comparison_group_for_stage(training_stage: Any) -> str:
    return "sft_rl" if str(training_stage or "").strip().lower() in {"sft", "rl"} else "base"


def _comparison_group_label(group: Any) -> str:
    return "SFT/RL Models" if str(group or "").strip().lower() == "sft_rl" else "Base Models"


def _row_prompt_version(row: Dict[str, Any]) -> str:
    prompt_version = str(row.get("prompt_version") or "").strip()
    if prompt_version:
        return prompt_version
    return _infer_prompt_version(row.get("file"), row.get("model"))


def _filter_rows_by_prompt_version(rows: List[Dict[str, Any]], prompt_version: str) -> List[Dict[str, Any]]:
    return [row for row in rows if _row_prompt_version(row) == prompt_version]


def _weighted_mean_from_rows(rows: List[Dict[str, Any]], value_key: str, weight_key: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        weight = _to_float(row.get(weight_key))
        if weight <= 0:
            continue
        numerator += _to_float(row.get(value_key)) * weight
        denominator += weight
    return _safe_div(numerator, denominator)


def _select_reference_prompt_version(rows: List[Dict[str, Any]]) -> Optional[str]:
    if not rows:
        return None

    coverage: Dict[str, Set[str]] = defaultdict(set)
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        prompt_version = _row_prompt_version(row)
        model_name = str(row.get("model") or "").strip() or str(row.get("file") or "").strip()
        coverage[prompt_version].add(model_name)
        counts[prompt_version] += 1

    for preferred in ("simple-prompt", "complex-prompt"):
        if preferred in coverage:
            return preferred

    ranked = sorted(
        coverage.keys(),
        key=lambda key: (
            -len(coverage[key]),
            -counts[key],
            PROMPT_VERSION_PRIORITY.get(key, 99),
            key,
        ),
    )
    return ranked[0] if ranked else None


def _available_prompt_versions(rows: List[Dict[str, Any]]) -> List[str]:
    present = {_row_prompt_version(row) for row in rows}
    return [prompt_version for prompt_version in ("simple-prompt", "complex-prompt") if prompt_version in present]


def _infer_model_label(path: Path) -> str:
    stem = path.stem
    match = re.search(r"builtin_function_call_(.+)$", stem)
    if match:
        return _strip_prompt_version_suffix(match.group(1))
    prefixes = ["test_openai_", "test_vllm_", "test_"]
    for prefix in prefixes:
        if stem.startswith(prefix):
            return _strip_prompt_version_suffix(stem[len(prefix) :])
    return _strip_prompt_version_suffix(stem)


def _model_display_sort_key(model: Any) -> Tuple[int, int, int, str]:
    """
    Order models for display:
    1) Qwen 4B (raw, SFT, RL)
    2) Qwen 8B (raw, SFT, RL)
    3) Qwen 32B (raw, SFT, RL)
    4) gpt-4.1-mini
    5) gpt-4.1
    6) gpt-5.1
    7) Gemini flash-lite
    8) Gemini flash
    9) Gemini pro
    Unknown models appear after known ones (alphabetical).
    """
    model_str = _strip_prompt_version_suffix(model)
    lower = model_str.lower()
    unified = lower.replace("_", "-")
    compact = re.sub(r"[^a-z0-9]+", "", lower)

    # Stage order for Qwen variants: raw -> SFT -> RL
    training_stage = _infer_training_stage(model_str)
    if training_stage == "sft":
        stage_rank = 1
    elif training_stage == "rl":
        stage_rank = 2
    else:
        stage_rank = 0

    is_qwen = "qwen" in unified and "vl" in unified
    if is_qwen:
        if "4b" in unified:
            return (0, 0, stage_rank, lower)
        if "8b" in unified:
            return (0, 1, stage_rank, lower)
        if "32b" in unified:
            return (0, 2, stage_rank, lower)
        return (0, 99, stage_rank, lower)

    if "gpt41mini" in compact:
        return (0, 3, 0, lower)
    if "gpt41" in compact:
        return (0, 4, 0, lower)
    if "gpt51" in compact:
        return (0, 5, 0, lower)
    if "gemini" in compact:
        if "flashlite" in compact:
            return (0, 6, 0, lower)
        if re.search(r"(?:^|-)flash(?:$|-)", unified):
            return (0, 7, 0, lower)
        if re.search(r"(?:^|-)pro(?:$|-)", unified):
            return (0, 8, 0, lower)
        return (0, 9, 0, lower)

    return (1, 999, 0, lower)


def _is_tool_rename_artifact(*, model: Any, file_name: Any) -> bool:
    lower_model = str(model or "").lower()
    lower_file = str(file_name or "").lower()
    return (
        "tool_rename" in lower_model
        or "too_rename" in lower_model
        or "tool_rename" in lower_file
        or "too_rename" in lower_file
        or "verb_alias_multi_v1" in lower_model
        or "verb_alias_multi_v1" in lower_file
    )


def _contains_sft_3e(*values: Any) -> bool:
    combined = " ".join(str(value or "") for value in values).strip().lower()
    unified = re.sub(r"[^a-z0-9]+", "-", combined).strip("-")
    return bool(re.search(r"(?:^|-)sft-(?:3e|3epoch(?:-rerun)?)(?:$|-)", unified))


def _normalized_model_tokens(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")


def _is_qwen_4b_instruct_target(model: Any) -> bool:
    unified_model = _normalized_model_tokens(model)
    return "qwen3-vl-4b-instruct" in unified_model


def _tool_rename_sft_label(*values: Any) -> str:
    return "SFT-3e" if _contains_sft_3e(*values) else "SFT"


def _is_exact_qwen_4b_instruct_base_model(model: Any) -> bool:
    return _normalized_model_tokens(model) == "qwen3-vl-4b-instruct"


def _is_exact_qwen_4b_instruct_sft_model(model: Any) -> bool:
    unified_model = _normalized_model_tokens(model)
    return bool(re.fullmatch(r"qwen3-vl-4b-instruct-sft(?:-3e|-3epoch(?:-rerun)?)?", unified_model))


def _is_exact_qwen_4b_instruct_sft_tool_rename_model(model: Any) -> bool:
    unified_model = _normalized_model_tokens(model)
    return bool(
        re.fullmatch(
            r"qwen3-vl-4b-instruct-sft(?:-3e|-3epoch(?:-rerun)?)?-(?:tool|too)-rename",
            unified_model,
        )
    )


def _classify_qwen_4b_variant(model: Any, file_name: Any) -> Optional[Tuple[str, bool, int, str]]:
    model_str = str(model or "").strip()
    lower_model = model_str.lower()
    unified = lower_model.replace("_", "-")
    if not ("qwen" in unified and "vl" in unified and "4b" in unified):
        return None

    if "sft" in unified:
        stage = "sft"
    elif "rl" in unified or "grpo" in unified:
        stage = "rl"
    else:
        stage = "raw"

    is_tool_rename = _is_tool_rename_artifact(model=model, file_name=file_name)
    label_map = {
        ("raw", False): "Raw",
        ("sft", False): "SFT",
        ("rl", False): "RL-GRPO",
        ("raw", True): "Raw + Tool Rename",
        ("sft", True): "SFT + Tool Rename",
        ("rl", True): "RL-GRPO + Tool Rename",
    }
    order_map = {
        ("raw", False): 0,
        ("sft", False): 1,
        ("rl", False): 2,
        ("raw", True): 3,
        ("sft", True): 4,
        ("rl", True): 5,
    }
    key = (stage, is_tool_rename)
    return stage, is_tool_rename, order_map[key], label_map[key]


def _classify_qwen_4b_sft_tool_rename_target(
    model: Any,
    file_name: Any,
) -> Optional[Tuple[str, str, bool, int, str]]:
    if not _is_qwen_4b_instruct_target(model):
        return None

    training_stage = _infer_training_stage(model, file_name)
    is_tool_rename = _is_tool_rename_artifact(model=model, file_name=file_name)
    if is_tool_rename:
        if training_stage != "sft":
            return None
        sft_label = _tool_rename_sft_label(model, file_name)
        return ("sft", "sft_tool_rename", True, 2, f"{sft_label} + Tool Rename")

    if training_stage == "base":
        return ("base", "base", False, 0, "Base")
    if training_stage == "sft":
        return ("sft", "sft_baseline", False, 1, _tool_rename_sft_label(model, file_name))
    return None


def _parse_question_id(record: Dict[str, Any]) -> str:
    for key in ("record_id", "id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    uid = _uid_base(record.get("uid"))
    if uid:
        return uid
    return ""


def _uid_base(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split(":", 1)[0]


def _parse_traj_id(record: Dict[str, Any]) -> int:
    uid = str(record.get("uid") or "")
    match = TRAJ_SUFFIX_RE.search(uid)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return 0
    return 0


def _parse_function_call_payload(value: Any) -> Tuple[str, Any]:
    if isinstance(value, dict):
        if "name" in value:
            return str(value.get("name") or "").strip(), value.get("arguments")
        if "function" in value and isinstance(value.get("function"), dict):
            fn = value["function"]
            return str(fn.get("name") or "").strip(), fn.get("arguments")
        return "", value.get("arguments")
    if isinstance(value, str):
        parsed = _parse_json_like(value)
        if isinstance(parsed, dict):
            return _parse_function_call_payload(parsed)
        return "", None
    return "", None


def _resolve_ground(tool_schema_mode: Any, tool_call_mode: Any) -> Optional[str]:
    schema = str(tool_schema_mode or "").strip()
    call_mode = str(tool_call_mode or "").strip()

    exact = GROUND_BY_MODE.get((schema, call_mode))
    if exact is not None:
        return exact

    # Support builtin-function-call variants such as:
    # builtin_function_call_verb_alias_multi_v1
    if call_mode.startswith("builtin_function_call"):
        if schema == "all_tools":
            return "all-tools"
        if schema == "dataset_mapped":
            return "mapped-build-in-functions"
    return None


def _is_error_observation(value: Any) -> Tuple[bool, str]:
    if value is None:
        return True, "missing_observation"
    if isinstance(value, dict):
        if value.get("error"):
            return True, "dict_error"
        return False, ""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return True, "empty_observation"
        if text.lower().startswith("error"):
            return True, "error_prefix"
        parsed = _parse_json_like(text)
        if isinstance(parsed, dict) and parsed.get("error"):
            return True, "json_error"
        return False, ""
    return False, ""


def _parse_trajectory_steps(conversations: Any) -> List[TrajectoryStep]:
    items = conversations if isinstance(conversations, list) else []
    steps: List[TrajectoryStep] = []
    pending_indices: deque[int] = deque()

    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("from") or "").strip().lower()
        if role == "function_call":
            name, arguments_raw = _parse_function_call_payload(item.get("value"))
            step = TrajectoryStep(
                step_index=len(steps),
                function_name=name,
                arguments_raw=arguments_raw,
                observation_raw=None,
                arguments_norm=_normalize_value(arguments_raw),
                observation_norm=None,
                is_error=False,
                error_type="",
            )
            steps.append(step)
            pending_indices.append(step.step_index)
            continue

        if role == "observation" and pending_indices:
            step_idx = pending_indices.popleft()
            obs_raw = item.get("value")
            steps[step_idx].observation_raw = obs_raw
            steps[step_idx].observation_norm = _normalize_value(obs_raw)

    for idx in pending_indices:
        steps[idx].is_error = True
        steps[idx].error_type = "missing_observation"

    for step in steps:
        if step.is_error:
            continue
        is_err, err_type = _is_error_observation(step.observation_raw)
        step.is_error = is_err
        step.error_type = err_type

    return steps


def _extract_final_answer(conversations: Any) -> Tuple[str, bool]:
    if not isinstance(conversations, list) or not conversations:
        return "", True

    final_message = ""
    for item in reversed(conversations):
        if not isinstance(item, dict):
            continue
        role = str(item.get("from") or "").strip().lower()
        if role in {"gpt", "assistant", "model"}:
            value = item.get("value")
            final_message = str(value or "")
            break
    if not final_message and isinstance(conversations[-1], dict):
        final_message = str(conversations[-1].get("value") or "")

    text = final_message.strip()
    if not text:
        return "", True

    matches = list(FINAL_ANSWER_RE.finditer(text))
    if matches:
        answer = text[matches[-1].end() :].strip()
        return answer, False
    return text, True


def _tools_any_order(expected: List[str], actual: List[str]) -> bool:
    need = Counter(expected)
    have = Counter(actual)
    for name, count in need.items():
        if have.get(name, 0) < count:
            return False
    return True


def _tools_in_order(expected: List[str], actual: List[str]) -> bool:
    if not expected:
        return True
    idx = 0
    for tool_name in actual:
        if tool_name == expected[idx]:
            idx += 1
            if idx == len(expected):
                return True
    return False


def _compute_tool_metrics(
    *,
    expected_steps: List[ExpectedStep],
    actual_steps: List[TrajectoryStep],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    expected_names = [step.function_name for step in expected_steps]
    actual_names = [step.function_name for step in actual_steps]

    any_order = _tools_any_order(expected_names, actual_names)
    in_order = _tools_in_order(expected_names, actual_names)
    exact_match = expected_names == actual_names

    details: List[Dict[str, Any]] = []
    matched_expected_steps = 0
    cursor = 0

    for expected in expected_steps:
        matched_actual_idx: Optional[int] = None
        for actual_idx in range(cursor, len(actual_steps)):
            if actual_steps[actual_idx].function_name == expected.function_name:
                matched_actual_idx = actual_idx
                break

        if matched_actual_idx is None:
            details.append(
                {
                    "expected_step_index": expected.step_index,
                    "expected_tool": expected.function_name,
                    "actual_step_index": "",
                    "actual_tool": "",
                    "tool_name_match": 0,
                    "args_match": 0,
                    "output_match": 0,
                    "step_match": 0,
                    "expected_arguments_norm": _value_snippet(expected.arguments_norm),
                    "actual_arguments_norm": "",
                    "expected_output_norm": _value_snippet(expected.output_norm),
                    "actual_output_norm": "",
                    "expected_arguments_norm_full": expected.arguments_norm,
                    "actual_arguments_norm_full": None,
                }
            )
            continue

        actual = actual_steps[matched_actual_idx]
        cursor = matched_actual_idx + 1
        args_match = _values_match(expected.arguments_norm, actual.arguments_norm)
        output_match = _values_match(expected.output_norm, actual.observation_norm)
        step_match = args_match and output_match
        if step_match:
            matched_expected_steps += 1

        details.append(
            {
                "expected_step_index": expected.step_index,
                "expected_tool": expected.function_name,
                "actual_step_index": matched_actual_idx,
                "actual_tool": actual.function_name,
                "tool_name_match": 1,
                "args_match": int(args_match),
                "output_match": int(output_match),
                "step_match": int(step_match),
                "expected_arguments_norm": _value_snippet(expected.arguments_norm),
                "actual_arguments_norm": _value_snippet(actual.arguments_norm),
                "expected_output_norm": _value_snippet(expected.output_norm),
                "actual_output_norm": _value_snippet(actual.observation_norm),
                "expected_arguments_norm_full": expected.arguments_norm,
                "actual_arguments_norm_full": actual.arguments_norm,
            }
        )

    if not expected_steps:
        parameters_score = 1.0
    else:
        parameters_score = matched_expected_steps / len(expected_steps)

    metrics = {
        "tools_any_order": int(any_order),
        "tools_in_order": int(in_order),
        "tools_exact_match": int(exact_match),
        "parameters_score": parameters_score,
        "expected_steps": len(expected_steps),
        "actual_steps": len(actual_steps),
        "parameter_matched_expected_steps": matched_expected_steps,
    }
    return metrics, details


def _evaluate_single_question(
    *,
    file_ground: str,
    model: str,
    file_name: str,
    question_id: str,
    question_type: str,
    question_type_bucket: str,
    traj_map: Dict[int, TrajectoryRecord],
    gt_item: Optional[GroundTruthItem],
    semantic_matcher: SemanticMatcher,
    max_traj: int,
    include_empty_questions: bool = False,
) -> QuestionEvaluation:
    if gt_item is None:
        return QuestionEvaluation(
            question_id=question_id,
            missing_gt=True,
            question_row=None,
            trajectory_rows=[],
            step_rows=[],
        )

    available_ids = sorted(tid for tid in traj_map.keys() if 0 <= tid < max_traj)
    if not available_ids:
        if include_empty_questions:
            question_text = gt_item.question or ""
            question_row = {
                "ground": file_ground,
                "model": model,
                "file": file_name,
                "question_id": question_id,
                "question_type": question_type,
                "question_type_bucket": question_type_bucket,
                "question": question_text,
                "gt_answer": gt_item.gt_answer,
                "reference_steps": len(gt_item.expected_steps),
                "available_traj_ids": "",
                "consumed_traj_ids": "",
                "tries_used": 0,
                "first_correct_traj": "",
                "accuracy": 0,
                "pass@1": 0,
                "pass@2": 0,
                "pass@3": 0,
                "consumed_tool_calls": 0,
                "consumed_function_call_errors": 0,
                "consumed_missing_final_answer_markers": 0,
            }
            return QuestionEvaluation(
                question_id=question_id,
                missing_gt=False,
                question_row=question_row,
                trajectory_rows=[],
                step_rows=[],
            )
        return QuestionEvaluation(
            question_id=question_id,
            missing_gt=False,
            question_row=None,
            trajectory_rows=[],
            step_rows=[],
        )

    expected_steps = gt_item.expected_steps
    gt_answer = gt_item.gt_answer
    question_text = gt_item.question or traj_map[available_ids[0]].question

    per_traj_tool_metrics: Dict[int, Dict[str, Any]] = {}
    per_traj_step_details: Dict[int, List[Dict[str, Any]]] = {}

    for traj_id in available_ids:
        traj = traj_map[traj_id]
        tool_metrics, details = _compute_tool_metrics(
            expected_steps=expected_steps,
            actual_steps=traj.steps,
        )
        per_traj_tool_metrics[traj_id] = tool_metrics
        per_traj_step_details[traj_id] = details

    consumed_ids: List[int] = []
    judged: Dict[int, Dict[str, Any]] = {}
    first_correct_traj: Optional[int] = None

    for traj_id in range(max_traj):
        traj = traj_map.get(traj_id)
        if traj is None:
            continue
        consumed_ids.append(traj_id)
        is_correct, reason, cache_hit, api_called = semantic_matcher.judge(
            question=question_text,
            gt_answer=gt_answer,
            pred_answer=traj.final_answer,
        )
        judged[traj_id] = {
            "is_correct": bool(is_correct),
            "reason": reason,
            "cache_hit": bool(cache_hit),
            "api_called": bool(api_called),
        }
        if is_correct:
            first_correct_traj = traj_id
            break

    pass_1 = int(first_correct_traj is not None and first_correct_traj < 1)
    pass_2 = int(first_correct_traj is not None and first_correct_traj < 2)
    pass_3 = int(first_correct_traj is not None and first_correct_traj < 3)
    accuracy = int(first_correct_traj is not None)

    consumed_tool_calls = sum(traj_map[tid].function_calls for tid in consumed_ids)
    consumed_errors = sum(traj_map[tid].function_call_errors for tid in consumed_ids)
    consumed_missing_markers = sum(
        1 for tid in consumed_ids if traj_map[tid].final_answer_missing_marker
    )

    question_row = {
        "ground": file_ground,
        "model": model,
        "file": file_name,
        "question_id": question_id,
        "question_type": question_type,
        "question_type_bucket": question_type_bucket,
        "question": question_text,
        "gt_answer": gt_answer,
        "reference_steps": len(expected_steps),
        "available_traj_ids": ",".join(str(x) for x in available_ids),
        "consumed_traj_ids": ",".join(str(x) for x in consumed_ids),
        "tries_used": len(consumed_ids),
        "first_correct_traj": "" if first_correct_traj is None else first_correct_traj,
        "accuracy": accuracy,
        "pass@1": pass_1,
        "pass@2": pass_2,
        "pass@3": pass_3,
        "consumed_tool_calls": consumed_tool_calls,
        "consumed_function_call_errors": consumed_errors,
        "consumed_missing_final_answer_markers": consumed_missing_markers,
    }

    trajectory_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []

    for traj_id in available_ids:
        traj = traj_map[traj_id]
        tool_metrics = per_traj_tool_metrics[traj_id]
        judge_info = judged.get(traj_id)
        consumed = traj_id in consumed_ids
        judged_flag = judge_info is not None
        is_correct_val: Any = ""
        semantic_reason = ""
        semantic_cache_hit = 0
        semantic_api_called = 0
        if judge_info is not None:
            is_correct_val = int(bool(judge_info.get("is_correct")))
            semantic_reason = str(judge_info.get("reason") or "")
            semantic_cache_hit = int(bool(judge_info.get("cache_hit")))
            semantic_api_called = int(bool(judge_info.get("api_called")))

        trajectory_rows.append(
            {
                "ground": file_ground,
                "model": model,
                "file": file_name,
                "question_id": question_id,
                "question_type": question_type,
                "question_type_bucket": question_type_bucket,
                "traj_id": traj_id,
                "consumed": int(consumed),
                "judged": int(judged_flag),
                "is_correct": is_correct_val,
                "semantic_reason": semantic_reason,
                "semantic_cache_hit": semantic_cache_hit,
                "semantic_api_called": semantic_api_called,
                "function_calls": traj.function_calls,
                "function_call_errors": traj.function_call_errors,
                "function_call_error_rate": _safe_div(traj.function_call_errors, traj.function_calls),
                "final_answer": traj.final_answer,
                "final_answer_missing_marker": int(traj.final_answer_missing_marker),
                "tools_any_order": tool_metrics["tools_any_order"],
                "tools_in_order": tool_metrics["tools_in_order"],
                "tools_exact_match": tool_metrics["tools_exact_match"],
                "parameters_score": tool_metrics["parameters_score"],
                "expected_steps": tool_metrics["expected_steps"],
                "actual_steps": tool_metrics["actual_steps"],
                "parameter_matched_expected_steps": tool_metrics["parameter_matched_expected_steps"],
            }
        )

        for detail in per_traj_step_details[traj_id]:
            step_rows.append(
                {
                    "ground": file_ground,
                    "model": model,
                    "file": file_name,
                    "question_id": question_id,
                    "question_type": question_type,
                    "question_type_bucket": question_type_bucket,
                    "traj_id": traj_id,
                    "consumed": int(consumed),
                    "judged": int(judged_flag),
                    "is_correct": is_correct_val,
                    **detail,
                }
            )

    return QuestionEvaluation(
        question_id=question_id,
        missing_gt=False,
        question_row=question_row,
        trajectory_rows=trajectory_rows,
        step_rows=step_rows,
    )


def _evaluate_prediction_file(
    *,
    path: Path,
    gt_map: Dict[str, GroundTruthItem],
    uid_question_type_map: Dict[str, Dict[str, str]],
    semantic_matcher: SemanticMatcher,
    max_traj: int,
    max_questions: Optional[int],
    semantic_workers: int,
    show_progress: bool,
    min_function_calls: int = 0,
    include_empty_questions: bool = False,
) -> Optional[FileEvaluation]:
    model = _infer_model_label(path)
    prompt_version = _infer_prompt_version(path.name, path.stem)

    per_question: Dict[str, Dict[int, TrajectoryRecord]] = defaultdict(dict)
    question_type_by_question: Dict[str, str] = {}
    question_type_bucket_by_question: Dict[str, str] = {}
    all_question_ids: Set[str] = set()
    ignored_question_ids: Set[str] = set()
    file_ground: Optional[str] = None

    for _line_no, record in iter_jsonl(path):
        ground = _resolve_ground(
            record.get("tool_schema_mode"),
            record.get("tool_call_mode"),
        )
        if ground is None:
            continue
        if file_ground is None:
            file_ground = ground
        if ground != file_ground:
            # Mixed mode file is unexpected; keep first mode only.
            continue

        question_id = _parse_question_id(record)
        if not question_id:
            continue
        traj_id = _parse_traj_id(record)
        if traj_id < 0 or traj_id >= max_traj:
            continue
        all_question_ids.add(question_id)

        uid_base = _uid_base(record.get("uid")) or question_id
        question_type_meta = uid_question_type_map.get(uid_base) or uid_question_type_map.get(question_id) or {}
        raw_question_type = question_type_meta.get("question_type") or record.get("question_type")
        raw_data_source = question_type_meta.get("data_source") or record.get("data_source")
        mapped_question_type, mapped_question_type_bucket = resolve_question_type_and_bucket(
            raw_question_type,
            raw_data_source,
            question_type_bucket_map=QUESTION_TYPE_BUCKET_MAP,
            apply_xbd_alias=True,
            unknown_value="unknown",
        )
        if _should_ignore_question_type(mapped_question_type):
            ignored_question_ids.add(question_id)
            continue
        current_question_type = question_type_by_question.get(question_id, "")
        if not current_question_type and mapped_question_type:
            question_type_by_question[question_id] = mapped_question_type
        current_question_type_bucket = question_type_bucket_by_question.get(question_id, "")
        if not current_question_type_bucket and mapped_question_type_bucket:
            question_type_bucket_by_question[question_id] = mapped_question_type_bucket

        conversations = record.get("conversations")
        steps = _parse_trajectory_steps(conversations)
        if len(steps) < max(0, int(min_function_calls)):
            continue
        final_answer, missing_marker = _extract_final_answer(conversations)
        question_text = str(record.get("question") or "").strip()

        traj = TrajectoryRecord(
            ground=ground,
            model=model,
            file=path.name,
            question_id=question_id,
            question=question_text,
            traj_id=traj_id,
            uid=str(record.get("uid") or ""),
            steps=steps,
            final_answer=final_answer,
            final_answer_missing_marker=missing_marker,
        )
        per_question[question_id][traj_id] = traj

    if file_ground is None:
        return None

    question_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    missing_gt_questions = 0

    if include_empty_questions:
        question_ids = sorted(all_question_ids)
    else:
        question_ids = sorted(per_question.keys())
    if ignored_question_ids:
        question_ids = [question_id for question_id in question_ids if question_id not in ignored_question_ids]
    if max_questions is not None:
        question_ids = question_ids[: max(0, max_questions)]

    if not question_ids:
        return FileEvaluation(
            ground=file_ground,
            model=model,
            file=path.name,
            prompt_version=prompt_version,
            question_rows=question_rows,
            trajectory_rows=trajectory_rows,
            step_rows=step_rows,
            missing_gt_questions=missing_gt_questions,
        )

    worker_count = max(1, int(semantic_workers))
    futures = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for question_id in question_ids:
            futures.append(
                executor.submit(
                    _evaluate_single_question,
                    file_ground=file_ground,
                    model=model,
                    file_name=path.name,
                    question_id=question_id,
                    question_type=question_type_by_question.get(question_id, ""),
                    question_type_bucket=question_type_bucket_by_question.get(question_id, ""),
                    traj_map=per_question.get(question_id, {}),
                    gt_item=gt_map.get(question_id),
                    semantic_matcher=semantic_matcher,
                    max_traj=max_traj,
                    include_empty_questions=include_empty_questions,
                )
            )

        completed_iter = as_completed(futures)
        if tqdm is not None:
            completed_iter = tqdm(
                completed_iter,
                total=len(futures),
                desc=f"{path.name} questions",
                leave=False,
                disable=not show_progress,
            )

        collected: List[QuestionEvaluation] = []
        for future in completed_iter:
            collected.append(future.result())

    collected.sort(key=lambda item: item.question_id)
    for item in collected:
        if item.missing_gt:
            missing_gt_questions += 1
            continue
        if item.question_row is not None:
            question_rows.append(item.question_row)
        trajectory_rows.extend(item.trajectory_rows)
        step_rows.extend(item.step_rows)

    return FileEvaluation(
        ground=file_ground,
        model=model,
        file=path.name,
        prompt_version=prompt_version,
        question_rows=question_rows,
        trajectory_rows=trajectory_rows,
        step_rows=step_rows,
        missing_gt_questions=missing_gt_questions,
    )


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = {k: row.get(k, "") for k in fieldnames}
            if "question_type_bucket" in fieldnames:
                output_row["question_type_bucket"] = question_type_bucket_display_name(
                    output_row.get("question_type_bucket", "")
                )
            writer.writerow(output_row)


def _display_question_type_bucket_labels(labels: List[str]) -> List[str]:
    return [question_type_bucket_display_name(label) for label in labels]


def _build_summary_rows(
    *,
    evaluations: List[FileEvaluation],
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []

    for eval_result in evaluations:
        q_rows = eval_result.question_rows
        t_rows = eval_result.trajectory_rows
        question_total = len(q_rows)
        training_stage = _infer_training_stage(eval_result.model, eval_result.file)

        if question_total == 0:
            summary_rows.append(
                {
                    "ground": eval_result.ground,
                    "model": eval_result.model,
                    "file": eval_result.file,
                    "prompt_version": eval_result.prompt_version,
                    "training_stage": training_stage,
                    "question_total": 0,
                    "accuracy": 0.0,
                    "pass@1": 0.0,
                    "pass@2": 0.0,
                    "pass@3": 0.0,
                    "avg_tries_used": 0.0,
                    "reference_steps_total": 0,
                    "tool_calls_consumed_total": 0,
                    "efficiency_consumed": 0.0,
                    "avg_tool_calls_consumed_per_question": 0.0,
                    "illegal_function_call_error_rate_consumed": 0.0,
                    "tools_any_order_rate_consumed": 0.0,
                    "tools_in_order_rate_consumed": 0.0,
                    "tools_exact_match_rate_consumed": 0.0,
                    "parameters_score_mean_consumed": 0.0,
                    "semantic_judged_trajectories": 0,
                    "semantic_cache_hits": 0,
                    "semantic_api_calls": 0,
                    "no_function_call_trajectories_consumed": 0,
                    "no_function_call_rate_consumed": 0.0,
                    "missing_final_answer_marker_consumed": 0,
                    "missing_gt_questions": eval_result.missing_gt_questions,
                }
            )
            continue

        accuracy = _mean([_to_float(row.get("accuracy")) for row in q_rows])
        pass1 = _mean([_to_float(row.get("pass@1")) for row in q_rows])
        pass2 = _mean([_to_float(row.get("pass@2")) for row in q_rows])
        pass3 = _mean([_to_float(row.get("pass@3")) for row in q_rows])
        avg_tries = _mean([_to_float(row.get("tries_used")) for row in q_rows])

        ref_steps_total = sum(_to_float(row.get("reference_steps")) for row in q_rows)
        calls_consumed_total = sum(_to_float(row.get("consumed_tool_calls")) for row in q_rows)
        errors_consumed_total = sum(_to_float(row.get("consumed_function_call_errors")) for row in q_rows)

        consumed_rows = [row for row in t_rows if _to_int(row.get("consumed")) == 1]
        no_call_consumed = sum(1 for row in consumed_rows if _to_int(row.get("function_calls")) == 0)

        def _rate(scope_rows: List[Dict[str, Any]], key: str) -> float:
            return _mean([_to_float(row.get(key)) for row in scope_rows]) if scope_rows else 0.0

        summary_rows.append(
            {
                "ground": eval_result.ground,
                "model": eval_result.model,
                "file": eval_result.file,
                "prompt_version": eval_result.prompt_version,
                "training_stage": training_stage,
                "question_total": question_total,
                "accuracy": accuracy,
                "pass@1": pass1,
                "pass@2": pass2,
                "pass@3": pass3,
                "avg_tries_used": avg_tries,
                "reference_steps_total": ref_steps_total,
                "tool_calls_consumed_total": calls_consumed_total,
                "efficiency_consumed": _safe_div(calls_consumed_total, ref_steps_total),
                "avg_tool_calls_consumed_per_question": _safe_div(calls_consumed_total, question_total),
                "illegal_function_call_error_rate_consumed": _safe_div(errors_consumed_total, calls_consumed_total),
                "tools_any_order_rate_consumed": _rate(consumed_rows, "tools_any_order"),
                "tools_in_order_rate_consumed": _rate(consumed_rows, "tools_in_order"),
                "tools_exact_match_rate_consumed": _rate(consumed_rows, "tools_exact_match"),
                "parameters_score_mean_consumed": _rate(consumed_rows, "parameters_score"),
                "semantic_judged_trajectories": sum(_to_int(row.get("judged")) for row in t_rows),
                "semantic_cache_hits": sum(_to_int(row.get("semantic_cache_hit")) for row in t_rows),
                "semantic_api_calls": sum(_to_int(row.get("semantic_api_called")) for row in t_rows),
                "no_function_call_trajectories_consumed": no_call_consumed,
                "no_function_call_rate_consumed": _safe_div(no_call_consumed, len(consumed_rows)),
                "missing_final_answer_marker_consumed": sum(
                    _to_int(row.get("consumed_missing_final_answer_markers")) for row in q_rows
                ),
                "missing_gt_questions": eval_result.missing_gt_questions,
            }
        )

    summary_rows.sort(
        key=lambda row: (
            _model_display_sort_key(row.get("model")),
            PROMPT_VERSION_PRIORITY.get(_row_prompt_version(row), 99),
            str(row.get("file") or ""),
        )
    )
    return summary_rows


def _build_trajectory_length_distribution(trajectory_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[(str(row.get("model") or ""), str(row.get("file") or ""))].append(row)

    for (model, file_name), items in grouped.items():
        scoped = [x for x in items if _to_int(x.get("consumed")) == 1]
        total = len(scoped)
        if total == 0:
            continue
        counter = Counter((_to_int(x.get("traj_id")), _to_int(x.get("function_calls"))) for x in scoped)
        for (traj_id, function_calls), count in sorted(counter.items()):
            rows.append(
                {
                    "ground": str(items[0].get("ground") or ""),
                    "model": model,
                    "file": file_name,
                    "traj_id": traj_id,
                    "function_calls": function_calls,
                    "count": count,
                    "proportion": _safe_div(count, total),
                }
            )
    rows.sort(
        key=lambda row: (
            _model_display_sort_key(row.get("model")),
            _to_int(row["traj_id"]),
            _to_int(row["function_calls"]),
        )
    )
    return rows


def _build_tries_used_distribution(question_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in question_rows:
        grouped[(str(row.get("model") or ""), str(row.get("file") or ""))].append(row)

    for (model, file_name), items in grouped.items():
        total = len(items)
        count_by_tries: Counter[int] = Counter()
        solved_by_tries: Counter[int] = Counter()
        for row in items:
            tries = _to_int(row.get("tries_used"))
            count_by_tries[tries] += 1
            solved_by_tries[tries] += _to_int(row.get("accuracy"))

        for tries_used in sorted(count_by_tries.keys()):
            count = count_by_tries[tries_used]
            solved_count = solved_by_tries[tries_used]
            rows.append(
                {
                    "ground": str(items[0].get("ground") or ""),
                    "model": model,
                    "file": file_name,
                    "tries_used": tries_used,
                    "count": count,
                    "solved_count": solved_count,
                    "unsolved_count": count - solved_count,
                    "proportion": _safe_div(count, total),
                }
            )

    rows.sort(key=lambda row: (_model_display_sort_key(row.get("model")), _to_int(row["tries_used"])))
    return rows


def _build_function_call_error_summary(question_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in question_rows:
        grouped[(str(row.get("model") or ""), str(row.get("file") or ""))].append(row)

    for (model, file_name), items in grouped.items():
        calls = sum(_to_float(row.get("consumed_tool_calls")) for row in items)
        errors = sum(_to_float(row.get("consumed_function_call_errors")) for row in items)
        rows.append(
            {
                "ground": str(items[0].get("ground") or ""),
                "model": model,
                "file": file_name,
                "function_calls": calls,
                "function_call_errors": errors,
                "function_call_error_rate": _safe_div(errors, calls),
            }
        )

    rows.sort(key=lambda row: (_model_display_sort_key(row.get("model")), str(row.get("file") or "")))
    return rows


def _build_no_function_call_summary(trajectory_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[(str(row.get("model") or ""), str(row.get("file") or ""))].append(row)

    for (model, file_name), items in grouped.items():
        scoped = [x for x in items if _to_int(x.get("consumed")) == 1]
        trajectory_total = len(scoped)
        no_calls = sum(1 for row in scoped if _to_int(row.get("function_calls")) == 0)
        rows.append(
            {
                "ground": str(items[0].get("ground") or ""),
                "model": model,
                "file": file_name,
                "trajectory_total": trajectory_total,
                "no_function_call_trajectories": no_calls,
                "no_function_call_rate": _safe_div(no_calls, trajectory_total),
            }
        )

    rows.sort(key=lambda row: (_model_display_sort_key(row.get("model")), str(row.get("file") or "")))
    return rows


def _build_tool_step_summary(trajectory_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[(str(row.get("model") or ""), str(row.get("file") or ""))].append(row)

    for (model, file_name), items in grouped.items():
        scoped = [x for x in items if _to_int(x.get("consumed")) == 1]
        rows.append(
            {
                "ground": str(items[0].get("ground") or ""),
                "model": model,
                "file": file_name,
                "trajectory_count": len(scoped),
                "tools_any_order_rate": _mean([_to_float(x.get("tools_any_order")) for x in scoped]),
                "tools_in_order_rate": _mean([_to_float(x.get("tools_in_order")) for x in scoped]),
                "tools_exact_match_rate": _mean([_to_float(x.get("tools_exact_match")) for x in scoped]),
                "parameters_score_mean": _mean([_to_float(x.get("parameters_score")) for x in scoped]),
            }
        )

    rows.sort(key=lambda row: (_model_display_sort_key(row.get("model")), str(row.get("file") or "")))
    return rows


def _build_question_type_breakdown(
    question_rows: List[Dict[str, Any]],
    trajectory_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    q_grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    t_grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in question_rows:
        key = (
            str(row.get("model") or ""),
            str(row.get("file") or ""),
            _question_type_bucket_from_row(row),
        )
        q_grouped[key].append(row)

    for row in trajectory_rows:
        key = (
            str(row.get("model") or ""),
            str(row.get("file") or ""),
            _question_type_bucket_from_row(row),
        )
        t_grouped[key].append(row)

    rows: List[Dict[str, Any]] = []
    for (model, file_name, question_type_bucket), q_items in q_grouped.items():
        if not q_items:
            continue

        t_items = t_grouped.get((model, file_name, question_type_bucket), [])
        question_total = len(q_items)
        ref_steps_total = sum(_to_float(row.get("reference_steps")) for row in q_items)
        calls_consumed_total = sum(_to_float(row.get("consumed_tool_calls")) for row in q_items)
        errors_consumed_total = sum(_to_float(row.get("consumed_function_call_errors")) for row in q_items)

        consumed_rows = [row for row in t_items if _to_int(row.get("consumed")) == 1]

        def _rate(scope_rows: List[Dict[str, Any]], key: str) -> float:
            return _mean([_to_float(row.get(key)) for row in scope_rows]) if scope_rows else 0.0

        rows.append(
            {
                "ground": str(q_items[0].get("ground") or ""),
                "model": model,
                "file": file_name,
                "question_type_bucket": question_type_bucket,
                "question_total": question_total,
                "accuracy": _mean([_to_float(row.get("accuracy")) for row in q_items]),
                "pass@1": _mean([_to_float(row.get("pass@1")) for row in q_items]),
                "pass@2": _mean([_to_float(row.get("pass@2")) for row in q_items]),
                "pass@3": _mean([_to_float(row.get("pass@3")) for row in q_items]),
                "avg_tries_used": _mean([_to_float(row.get("tries_used")) for row in q_items]),
                "reference_steps_total": ref_steps_total,
                "tool_calls_consumed_total": calls_consumed_total,
                "efficiency_consumed": _safe_div(calls_consumed_total, ref_steps_total),
                "illegal_function_call_error_rate_consumed": _safe_div(errors_consumed_total, calls_consumed_total),
                "tools_any_order_rate_consumed": _rate(consumed_rows, "tools_any_order"),
                "tools_in_order_rate_consumed": _rate(consumed_rows, "tools_in_order"),
                "tools_exact_match_rate_consumed": _rate(consumed_rows, "tools_exact_match"),
                "parameters_score_mean_consumed": _rate(consumed_rows, "parameters_score"),
            }
        )

    rows.sort(
        key=lambda row: (
            _model_display_sort_key(row.get("model")),
            question_type_bucket_sort_key(row.get("question_type_bucket")),
            str(row.get("file") or ""),
        )
    )
    return rows


def _aggregate_question_type_bucket_model_passk_rows(
    question_rows: List[Dict[str, Any]],
    models: List[str],
) -> List[Dict[str, Any]]:
    if not question_rows or not models:
        return []

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    prompt_versions = {_row_prompt_version(row) for row in question_rows if _row_prompt_version(row)}
    prompt_version = prompt_versions.pop() if len(prompt_versions) == 1 else "mixed"

    for row in question_rows:
        grouped[(_question_type_bucket_from_row(row), str(row.get("model") or ""))].append(row)

    question_type_buckets = sorted(
        {_question_type_bucket_from_row(row) for row in question_rows},
        key=question_type_bucket_sort_key,
    )
    rows: List[Dict[str, Any]] = []
    for question_type_bucket in question_type_buckets:
        for model in models:
            scoped = grouped.get((question_type_bucket, model), [])
            rows.append(
                {
                    "prompt_version": prompt_version,
                    "question_type_bucket": question_type_bucket,
                    "model": model,
                    "question_total": len(scoped),
                    "pass@1": _mean([_to_float(item.get("pass@1")) for item in scoped]) if scoped else 0.0,
                    "pass@2": _mean([_to_float(item.get("pass@2")) for item in scoped]) if scoped else 0.0,
                    "pass@3": _mean([_to_float(item.get("pass@3")) for item in scoped]) if scoped else 0.0,
                }
            )
    return rows


def _build_counting_focus_analysis(
    *,
    question_rows: List[Dict[str, Any]],
    trajectory_rows: List[Dict[str, Any]],
    step_rows: List[Dict[str, Any]],
    parameter_semantic_matcher: ParameterSemanticMatcher,
    semantic_workers: int,
    show_progress: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    counting_bucket = "counting_arithmetic"
    counting_question_rows = [row for row in question_rows if _question_type_bucket_from_row(row) == counting_bucket]
    counting_trajectory_rows = [row for row in trajectory_rows if _question_type_bucket_from_row(row) == counting_bucket]

    if not counting_trajectory_rows:
        return [], []

    question_text_by_key: Dict[Tuple[str, str, str], str] = {}
    for row in counting_question_rows:
        key = (
            str(row.get("model") or ""),
            str(row.get("file") or ""),
            str(row.get("question_id") or ""),
        )
        if key not in question_text_by_key:
            question_text_by_key[key] = str(row.get("question") or "")

    candidate_step_rows: List[Dict[str, Any]] = []
    for row in step_rows:
        if _question_type_bucket_from_row(row) != counting_bucket:
            continue
        if _to_int(row.get("tool_name_match")) != 1:
            continue
        candidate_step_rows.append(row)

    detail_rows: List[Dict[str, Any]] = []
    if candidate_step_rows:
        worker_count = max(1, int(semantic_workers))

        def _judge_row(row: Dict[str, Any]) -> Dict[str, Any]:
            model = str(row.get("model") or "")
            file_name = str(row.get("file") or "")
            question_id = str(row.get("question_id") or "")
            question_text = question_text_by_key.get((model, file_name, question_id), "")
            expected_args = row.get("expected_arguments_norm_full")
            actual_args = row.get("actual_arguments_norm_full")
            expected_tool = str(row.get("expected_tool") or "")

            args_semantic_same, semantic_reason, cache_hit, api_called = parameter_semantic_matcher.judge(
                question=question_text,
                tool_name=expected_tool,
                reference_arguments=expected_args,
                candidate_arguments=actual_args,
            )

            return {
                "ground": str(row.get("ground") or ""),
                "model": model,
                "file": file_name,
                "question_id": question_id,
                "question_type": str(row.get("question_type") or ""),
                "question_type_bucket": counting_bucket,
                "traj_id": _to_int(row.get("traj_id")),
                "consumed": _to_int(row.get("consumed")),
                "expected_step_index": _to_int(row.get("expected_step_index")),
                "expected_tool": expected_tool,
                "actual_step_index": _to_int(row.get("actual_step_index")),
                "actual_tool": str(row.get("actual_tool") or ""),
                "args_match": _to_int(row.get("args_match")),
                "args_semantic_same": int(bool(args_semantic_same)),
                "args_semantic_reason": str(semantic_reason or ""),
                "args_semantic_cache_hit": int(bool(cache_hit)),
                "args_semantic_api_called": int(bool(api_called)),
                "expected_arguments_norm": str(row.get("expected_arguments_norm") or ""),
                "actual_arguments_norm": str(row.get("actual_arguments_norm") or ""),
            }

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_judge_row, row) for row in candidate_step_rows]
            completed_iter = as_completed(futures)
            if tqdm is not None:
                completed_iter = tqdm(
                    completed_iter,
                    total=len(futures),
                    desc="Counting args semantic",
                    leave=False,
                    disable=not show_progress,
                )
            for future in completed_iter:
                detail_rows.append(future.result())

    detail_rows.sort(
        key=lambda row: (
            _model_display_sort_key(row.get("model")),
            str(row.get("file") or ""),
            str(row.get("question_id") or ""),
            _to_int(row.get("traj_id")),
            _to_int(row.get("expected_step_index")),
        )
    )

    grouped_traj: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    grouped_detail: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in counting_trajectory_rows:
        key = (
            str(row.get("ground") or ""),
            str(row.get("model") or ""),
            str(row.get("file") or ""),
        )
        grouped_traj[key].append(row)
    for row in detail_rows:
        key = (
            str(row.get("ground") or ""),
            str(row.get("model") or ""),
            str(row.get("file") or ""),
        )
        grouped_detail[key].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped_traj.keys(), key=lambda k: (_model_display_sort_key(k[1]), k[2])):
        ground, model, file_name = key
        traj_items = [row for row in grouped_traj.get(key, []) if _to_int(row.get("consumed")) == 1]
        detail_items = [row for row in grouped_detail.get(key, []) if _to_int(row.get("consumed")) == 1]

        counting_question_total = len({str(row.get("question_id") or "") for row in traj_items})
        summary_rows.append(
            {
                "ground": ground,
                "model": model,
                "file": file_name,
                "counting_question_total": counting_question_total,
                "counting_trajectory_total": len(traj_items),
                "right_tool_any_order_rate": _mean([_to_float(row.get("tools_any_order")) for row in traj_items]),
                "right_tool_in_order_rate": _mean([_to_float(row.get("tools_in_order")) for row in traj_items]),
                "right_tool_exact_match_rate": _mean([_to_float(row.get("tools_exact_match")) for row in traj_items]),
                "tool_name_matched_steps": len(detail_items),
                "args_exact_match_rate": _mean([_to_float(row.get("args_match")) for row in detail_items]),
                "args_semantic_same_rate": _mean(
                    [_to_float(row.get("args_semantic_same")) for row in detail_items]
                ),
                "args_semantic_same_steps": sum(_to_int(row.get("args_semantic_same")) for row in detail_items),
                "args_semantic_cache_hits": sum(
                    _to_int(row.get("args_semantic_cache_hit")) for row in detail_items
                ),
                "args_semantic_api_calls": sum(
                    _to_int(row.get("args_semantic_api_called")) for row in detail_items
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            _model_display_sort_key(row.get("model")),
            str(row.get("file") or ""),
        )
    )
    return summary_rows, detail_rows


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        print(f"[WARN] Matplotlib unavailable, skipping charts: {exc}", file=sys.stderr)
        return None


def _format_bar_value(value: float, *, as_int: bool) -> str:
    if as_int:
        return str(int(round(value)))
    return f"{value:.2f}"


def _add_bar_value_labels(
    ax,
    bar_container,
    *,
    as_int: bool,
    label_type: str = "edge",
    rotation: float = 45.0,
    fontsize: int = 7,
) -> None:
    labels: List[str] = []
    for bar in bar_container:
        value = float(bar.get_height())
        if not math.isfinite(value):
            labels.append("")
            continue
        labels.append(_format_bar_value(value, as_int=as_int))

    try:
        ax.bar_label(
            bar_container,
            labels=labels,
            label_type=label_type,
            padding=2,
            fontsize=fontsize,
            rotation=rotation,
        )
        return
    except Exception:
        # Fallback for older matplotlib builds without reliable bar_label support.
        pass

    y_max = float(ax.get_ylim()[1]) if ax.get_ylim() else 1.0
    offset = max(0.01 * y_max, 1e-6)
    for bar, label in zip(bar_container, labels):
        if not label:
            continue
        x = bar.get_x() + bar.get_width() / 2.0
        if label_type == "center":
            y = bar.get_y() + bar.get_height() / 2.0
            va = "center"
        else:
            if bar.get_height() >= 0:
                y = bar.get_y() + bar.get_height() + offset
                va = "bottom"
            else:
                y = bar.get_y() + bar.get_height() - offset
                va = "top"
        ax.text(x, y, label, ha="center", va=va, rotation=rotation, fontsize=fontsize)


def _plot_grouped_bars(
    plt,
    *,
    title: str,
    labels: List[str],
    series: List[Tuple[str, List[float]]],
    ylabel: str,
    output_path: Path,
    y_min: Optional[float] = 0.0,
    y_max: Optional[float] = None,
    draw_zero_line: bool = False,
) -> None:
    if not labels or not series:
        return

    fig_width = max(8.0, len(labels) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x_vals = list(range(len(labels)))
    width = min(0.75 / max(1, len(series)), 0.35)
    offsets = [(idx - (len(series) - 1) / 2) * width for idx in range(len(series))]

    max_val = 0.0
    min_val = 0.0
    bar_containers = []
    for idx, (name, values) in enumerate(series):
        positions = [x + offsets[idx] for x in x_vals]
        bar_container = ax.bar(positions, values, width=width, label=name)
        bar_containers.append(bar_container)
        if values:
            max_val = max(max_val, max(values))
            min_val = min(min_val, min(values))

    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    lower_bound = min_val if y_min is None else min(min_val, y_min)
    upper_bound = max_val if y_max is None else max(max_val, y_max)
    if lower_bound >= 0:
        upper_limit = max(1.0, upper_bound * 1.15)
        ax.set_ylim(0 if y_min is not None else lower_bound, upper_limit)
    else:
        span = max(1e-6, upper_bound - lower_bound)
        padding = max(0.08 * span, 0.02)
        ax.set_ylim(lower_bound - padding, upper_bound + padding)
        if draw_zero_line:
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    for bar_container in bar_containers:
        _add_bar_value_labels(
            ax,
            bar_container,
            as_int=False,
            label_type="edge",
            rotation=45.0,
            fontsize=6,
        )
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_question_type_passk_heatmaps(
    plt,
    *,
    output_dir: Path,
    plot_rows: List[Dict[str, Any]],
    models: List[str],
    file_suffix: str = "",
) -> None:
    if not plot_rows or not models:
        return

    question_types = sorted(
        {str(row.get("question_type_bucket") or "") for row in plot_rows},
        key=question_type_bucket_sort_key,
    )
    if not question_types:
        return

    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in plot_rows:
        question_type_bucket = str(row.get("question_type_bucket") or "")
        model = str(row.get("model") or "")
        grouped[(question_type_bucket, model)] = row

    metric_defs = [
        ("Pass@1", "pass@1"),
        ("Pass@2", "pass@2"),
        ("Pass@3", "pass@3"),
    ]

    fig_width = max(15.0, len(models) * 2.2)
    fig_height = max(8.0, len(question_types) * 0.45)
    fig, axes = plt.subplots(1, len(metric_defs), figsize=(fig_width, fig_height))
    axes_list = list(axes) if hasattr(axes, "__iter__") else [axes]

    for ax, (title, metric_key) in zip(axes_list, metric_defs):
        matrix: List[List[float]] = []
        for question_type_bucket in question_types:
            row_vals: List[float] = []
            for model in models:
                row = grouped.get((question_type_bucket, model), {})
                row_vals.append(_to_float(row.get(metric_key)))
            matrix.append(row_vals)

        image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks(list(range(len(models))))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(list(range(len(question_types))))
        ax.set_yticklabels(_display_question_type_bucket_labels(question_types), fontsize=8)

        for y_idx, row_vals in enumerate(matrix):
            for x_idx, val in enumerate(row_vals):
                text_color = "white" if val >= 0.6 else "black"
                ax.text(x_idx, y_idx, f"{val:.2f}", ha="center", va="center", fontsize=6, color=text_color)

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Question Type Bucket vs Model Pass@k Heatmaps", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"question_type_bucket_model_passk_heatmaps{file_suffix}.png", bbox_inches="tight")
    plt.close(fig)


def _plot_question_type_passk_grouped_bars(
    plt,
    *,
    output_dir: Path,
    plot_rows: List[Dict[str, Any]],
    models: List[str],
    file_suffix: str = "",
) -> None:
    if not plot_rows or not models:
        return

    question_types = sorted(
        {str(row.get("question_type_bucket") or "") for row in plot_rows},
        key=question_type_bucket_sort_key,
    )
    if not question_types:
        return

    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in plot_rows:
        question_type_bucket = str(row.get("question_type_bucket") or "")
        model = str(row.get("model") or "")
        grouped[(question_type_bucket, model)] = row

    metric_defs = [
        ("Pass@1", "pass@1"),
        ("Pass@2", "pass@2"),
        ("Pass@3", "pass@3"),
    ]

    fig_width = max(14.0, len(question_types) * 1.3)
    fig_height = max(10.0, len(metric_defs) * 3.2)
    fig, axes = plt.subplots(len(metric_defs), 1, figsize=(fig_width, fig_height), sharex=True)
    if len(metric_defs) == 1:
        axes = [axes]

    x_vals = list(range(len(question_types)))
    width = min(0.85 / max(1, len(models)), 0.18)
    offsets = [(idx - (len(models) - 1) / 2) * width for idx in range(len(models))]

    for ax, (title, metric_key) in zip(axes, metric_defs):
        for idx, model in enumerate(models):
            values: List[float] = []
            for question_type_bucket in question_types:
                row = grouped.get((question_type_bucket, model), {})
                values.append(_to_float(row.get(metric_key)))
            positions = [x + offsets[idx] for x in x_vals]
            bar_container = ax.bar(positions, values, width=width, label=model)
            _add_bar_value_labels(
                ax,
                bar_container,
                as_int=False,
                label_type="edge",
                rotation=45.0,
                fontsize=5,
            )

        ax.set_ylim(0, 1.15)
        ax.set_ylabel(title)
        ax.set_title(f"{title} by Question Type Bucket and Model")

    axes[-1].set_xticks(x_vals)
    axes[-1].set_xticklabels(
        _display_question_type_bucket_labels(question_types),
        rotation=35,
        ha="right",
        fontsize=8,
    )
    axes[0].legend(loc="upper left", ncol=min(4, len(models)), fontsize=7)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"question_type_bucket_model_passk_grouped_bars{file_suffix}.png", bbox_inches="tight")
    plt.close(fig)


def _generate_counting_focus_charts(
    *,
    output_dir: Path,
    counting_focus_summary_rows: List[Dict[str, Any]],
) -> None:
    if not counting_focus_summary_rows:
        return
    plt = _load_matplotlib()
    if plt is None:
        return

    scoped = sorted(counting_focus_summary_rows, key=lambda row: _model_display_sort_key(row.get("model")))
    labels = [str(row.get("model") or "") for row in scoped]
    _plot_grouped_bars(
        plt,
        title="Counting Questions: Right Tool + Argument Semantic Match",
        labels=labels,
        series=[
            ("Right Tool (Any Order)", [_to_float(row.get("right_tool_any_order_rate")) for row in scoped]),
            ("Right Tool (In Order)", [_to_float(row.get("right_tool_in_order_rate")) for row in scoped]),
            ("Args Semantic Same", [_to_float(row.get("args_semantic_same_rate")) for row in scoped]),
        ],
        ylabel="Rate",
        output_path=output_dir / "counting_focus_tool_and_args.png",
        y_max=1.0,
    )


def _generate_ground_charts(
    *,
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    question_rows: List[Dict[str, Any]],
    trajectory_rows: List[Dict[str, Any]],
    title_suffix: str = "",
    file_suffix: str = "",
) -> None:
    plt = _load_matplotlib()
    if plt is None:
        return

    if not summary_rows:
        return

    models = [str(row.get("model") or "") for row in summary_rows]

    _plot_grouped_bars(
        plt,
        title=f"Pass@k by Model{title_suffix}",
        labels=models,
        series=[
            ("Pass@1", [_to_float(row.get("pass@1")) for row in summary_rows]),
            ("Pass@2", [_to_float(row.get("pass@2")) for row in summary_rows]),
            ("Pass@3", [_to_float(row.get("pass@3")) for row in summary_rows]),
        ],
        ylabel="Rate",
        output_path=output_dir / f"accuracy_passk_by_model{file_suffix}.png",
        y_max=1.0,
    )

    _plot_grouped_bars(
        plt,
        title=f"Efficiency by Model{title_suffix}",
        labels=models,
        series=[
            ("Efficiency", [_to_float(row.get("efficiency_consumed")) for row in summary_rows]),
        ],
        ylabel="Efficiency",
        output_path=output_dir / f"efficiency_by_model{file_suffix}.png",
        y_max=None,
    )

    _plot_grouped_bars(
        plt,
        title=f"Illegal Function-Call Error Rate by Model{title_suffix}",
        labels=models,
        series=[
            ("Error Rate", [_to_float(row.get("illegal_function_call_error_rate_consumed")) for row in summary_rows]),
        ],
        ylabel="Error Rate",
        output_path=output_dir / f"illegal_call_error_rate_by_model{file_suffix}.png",
        y_max=1.0,
    )

    # trajectory_length_hist.png
    consumed_calls = [_to_int(row.get("function_calls")) for row in trajectory_rows if _to_int(row.get("consumed")) == 1]
    if consumed_calls:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = sorted(set(consumed_calls))
        counts = [consumed_calls.count(b) for b in bins]
        bar_container = ax.bar([str(x) for x in bins], counts)
        ax.set_title(f"Trajectory Length Distribution (Function Calls){title_suffix}")
        ax.set_xlabel("Function Calls")
        ax.set_ylabel("Trajectory Count")
        ax.set_ylim(0, max(1.0, max(counts) * 1.15))
        _add_bar_value_labels(ax, bar_container, as_int=True, label_type="edge", rotation=45.0, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"trajectory_length_hist{file_suffix}.png", bbox_inches="tight")
        plt.close(fig)

    # tries_used_hist.png
    if question_rows:
        tries_counter: Counter[int] = Counter(_to_int(row.get("tries_used")) for row in question_rows)
        solved_counter: Counter[int] = Counter()
        unsolved_counter: Counter[int] = Counter()
        for row in question_rows:
            tries = _to_int(row.get("tries_used"))
            if _to_int(row.get("accuracy")) == 1:
                solved_counter[tries] += 1
            else:
                unsolved_counter[tries] += 1
        x_vals = sorted(tries_counter.keys())
        solved_vals = [solved_counter[x] for x in x_vals]
        unsolved_vals = [unsolved_counter[x] for x in x_vals]

        fig, ax = plt.subplots(figsize=(8, 5))
        solved_bars = ax.bar([str(x) for x in x_vals], solved_vals, label="Solved")
        unsolved_bars = ax.bar([str(x) for x in x_vals], unsolved_vals, bottom=solved_vals, label="Unsolved")
        ax.set_title(f"Tries Used Distribution{title_suffix}")
        ax.set_xlabel("Tries Used")
        ax.set_ylabel("Question Count")
        stacked_max = max((s + u) for s, u in zip(solved_vals, unsolved_vals))
        ax.set_ylim(0, max(1.0, float(stacked_max) * 1.2))
        _add_bar_value_labels(ax, solved_bars, as_int=True, label_type="edge", rotation=45.0, fontsize=8)
        _add_bar_value_labels(ax, unsolved_bars, as_int=True, label_type="edge", rotation=45.0, fontsize=8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"tries_used_hist{file_suffix}.png", bbox_inches="tight")
        plt.close(fig)

    # passk_by_question_type_bucket.png
    if question_rows:
        by_question_type: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for row in question_rows:
            question_type_bucket = _question_type_bucket_from_row(row)
            by_question_type[question_type_bucket]["pass@1"].append(_to_float(row.get("pass@1")))
            by_question_type[question_type_bucket]["pass@2"].append(_to_float(row.get("pass@2")))
            by_question_type[question_type_bucket]["pass@3"].append(_to_float(row.get("pass@3")))
        labels = sorted(by_question_type.keys(), key=question_type_bucket_sort_key)
        pass1_vals = [_mean(by_question_type[label]["pass@1"]) for label in labels]
        pass2_vals = [_mean(by_question_type[label]["pass@2"]) for label in labels]
        pass3_vals = [_mean(by_question_type[label]["pass@3"]) for label in labels]
        _plot_grouped_bars(
            plt,
            title=f"Pass@k by Question Type Bucket{title_suffix}",
            labels=_display_question_type_bucket_labels(labels),
            series=[
                ("Pass@1", pass1_vals),
                ("Pass@2", pass2_vals),
                ("Pass@3", pass3_vals),
            ],
            ylabel="Rate",
            output_path=output_dir / f"accuracy_by_question_type_bucket{file_suffix}.png",
            y_max=1.0,
        )
        question_type_bucket_plot_rows = _aggregate_question_type_bucket_model_passk_rows(question_rows, models)
        _write_csv(
            output_dir / f"question_type_bucket_model_passk_heatmaps{file_suffix}.csv",
            QUESTION_TYPE_BUCKET_PLOT_FIELDNAMES,
            question_type_bucket_plot_rows,
        )
        _write_csv(
            output_dir / f"question_type_bucket_model_passk_grouped_bars{file_suffix}.csv",
            QUESTION_TYPE_BUCKET_PLOT_FIELDNAMES,
            question_type_bucket_plot_rows,
        )
        _plot_question_type_passk_heatmaps(
            plt,
            output_dir=output_dir,
            plot_rows=question_type_bucket_plot_rows,
            models=models,
            file_suffix=file_suffix,
        )
        _plot_question_type_passk_grouped_bars(
            plt,
            output_dir=output_dir,
            plot_rows=question_type_bucket_plot_rows,
            models=models,
            file_suffix=file_suffix,
        )

    # tool_step_rates_by_model.png
    fig_width = max(8.0, len(models) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    metric_pairs = [
        ("Any Order", "tools_any_order_rate"),
        ("In Order", "tools_in_order_rate"),
        ("Exact Match", "tools_exact_match_rate"),
        ("Param Mean", "parameters_score_mean"),
    ]

    x_vals = list(range(len(models)))
    width = 0.18
    offsets = [(-1.5 + i) * width for i in range(len(metric_pairs))]
    for idx, (label, key_base) in enumerate(metric_pairs):
        key = f"{key_base}_consumed"
        vals = [_to_float(row.get(key)) for row in summary_rows]
        pos = [x + offsets[idx] for x in x_vals]
        bar_container = ax.bar(pos, vals, width=width, label=label)
        _add_bar_value_labels(
            ax,
            bar_container,
            as_int=False,
            label_type="edge",
            rotation=45.0,
            fontsize=6,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.set_title(f"Tool-Step Rates{title_suffix}")
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    ax.set_xticks(list(range(len(models))))
    ax.set_xticklabels(models, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / f"tool_step_rates_by_model{file_suffix}.png", bbox_inches="tight")
    plt.close(fig)


def _build_ground_means(summary_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row.get("ground") or "unknown")].append(row)

    metrics = [
        "accuracy",
        "pass@1",
        "pass@2",
        "pass@3",
        "efficiency_consumed",
        "illegal_function_call_error_rate_consumed",
        "tools_any_order_rate_consumed",
        "tools_in_order_rate_consumed",
        "tools_exact_match_rate_consumed",
        "parameters_score_mean_consumed",
    ]

    out: Dict[str, Dict[str, float]] = {}
    for ground, rows in grouped.items():
        out[ground] = {}
        for metric in metrics:
            out[ground][metric] = _mean([_to_float(row.get(metric)) for row in rows])
    return out


def _tool_rename_target_sort_key(row: Dict[str, Any], variant: str) -> Tuple[int, int, str, str]:
    model = str(row.get("model") or "")
    file_name = str(row.get("file") or "")

    if variant == "base":
        exact_rank = int(not _is_exact_qwen_4b_instruct_base_model(model))
    elif variant == "sft_baseline":
        exact_rank = int(not _is_exact_qwen_4b_instruct_sft_model(model))
    else:
        exact_rank = int(not _is_exact_qwen_4b_instruct_sft_tool_rename_model(model))

    return (exact_rank, len(file_name), file_name, model)


def _build_tool_rename_comparison_rows(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped_by_ground: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped_by_ground[str(row.get("ground") or "unknown")].append(row)

    rows: List[Dict[str, Any]] = []
    required_variants = ("base", "sft_baseline", "sft_tool_rename")
    for ground, items in grouped_by_ground.items():
        selected_by_prompt: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for row in items:
            prompt_version = _row_prompt_version(row)
            if prompt_version not in {"simple-prompt", "complex-prompt"}:
                continue

            class_info = _classify_qwen_4b_sft_tool_rename_target(row.get("model"), row.get("file"))
            if class_info is None:
                continue

            stage, variant, is_tool_rename, order_idx, label = class_info
            existing = selected_by_prompt[prompt_version].get(variant)
            if existing is not None:
                existing_key = _tool_rename_target_sort_key(existing["row"], variant)
                candidate_key = _tool_rename_target_sort_key(row, variant)
                if candidate_key >= existing_key:
                    continue

            selected_by_prompt[prompt_version][variant] = {
                "row": row,
                "stage": stage,
                "variant": variant,
                "is_tool_rename": is_tool_rename,
                "comparison_order": order_idx,
                "comparison_label": label,
            }

        for prompt_version in ("simple-prompt", "complex-prompt"):
            selected = selected_by_prompt.get(prompt_version) or {}
            if not all(variant in selected for variant in required_variants):
                continue

            prompt_label = _prompt_version_display(prompt_version)
            sft_baseline = selected["sft_baseline"]["row"]
            for variant in required_variants:
                entry = selected[variant]
                row = entry["row"]
                baseline = sft_baseline if entry["is_tool_rename"] else None

                def _delta(metric: str) -> Any:
                    if baseline is None:
                        return ""
                    return _to_float(row.get(metric)) - _to_float(baseline.get(metric))

                rows.append(
                    {
                        "ground": ground,
                        "prompt_version": prompt_version,
                        "prompt_label": prompt_label,
                        "variant": variant,
                        "comparison_label": entry["comparison_label"],
                        "comparison_order": entry["comparison_order"],
                        "stage": entry["stage"],
                        "is_tool_rename": int(entry["is_tool_rename"]),
                        "model": str(row.get("model") or ""),
                        "file": str(row.get("file") or ""),
                        "question_total": _to_int(row.get("question_total")),
                        "pass@1": _to_float(row.get("pass@1")),
                        "pass@2": _to_float(row.get("pass@2")),
                        "pass@3": _to_float(row.get("pass@3")),
                        "accuracy": _to_float(row.get("accuracy")),
                        "efficiency_consumed": _to_float(row.get("efficiency_consumed")),
                        "no_function_call_trajectories_consumed": _to_int(
                            row.get("no_function_call_trajectories_consumed")
                        ),
                        "no_function_call_rate_consumed": _to_float(row.get("no_function_call_rate_consumed")),
                        "delta_pass@1_vs_stage_baseline": _delta("pass@1"),
                        "delta_pass@2_vs_stage_baseline": _delta("pass@2"),
                        "delta_pass@3_vs_stage_baseline": _delta("pass@3"),
                        "delta_accuracy_vs_stage_baseline": _delta("accuracy"),
                    }
                )

    rows.sort(
        key=lambda row: (
            str(row.get("ground") or ""),
            PROMPT_VERSION_PRIORITY.get(str(row.get("prompt_version") or ""), 99),
            _to_int(row.get("comparison_order")),
            str(row.get("file") or ""),
        )
    )
    return rows


def _generate_tool_rename_comparison_charts(*, chart_dir: Path, comparison_rows: List[Dict[str, Any]]) -> None:
    if not comparison_rows:
        return
    plt = _load_matplotlib()
    if plt is None:
        return

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        key = (
            str(row.get("ground") or "unknown"),
            str(row.get("prompt_version") or "unknown"),
        )
        grouped[key].append(row)

    chart_dir.mkdir(parents=True, exist_ok=True)
    for (ground, prompt_version), rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda row: (_to_int(row.get("comparison_order")), str(row.get("file") or "")))
        labels = [str(row.get("comparison_label") or "") for row in rows_sorted]
        prompt_label = _prompt_version_display(prompt_version)
        prompt_slug = prompt_version.replace("-", "_")
        _plot_grouped_bars(
            plt,
            title=f"Tool-Rename 4B Comparison ({ground}, {prompt_label})",
            labels=labels,
            series=[
                ("Pass@1", [_to_float(row.get("pass@1")) for row in rows_sorted]),
                ("Pass@2", [_to_float(row.get("pass@2")) for row in rows_sorted]),
                ("Pass@3", [_to_float(row.get("pass@3")) for row in rows_sorted]),
            ],
            ylabel="Rate",
            output_path=chart_dir / f"{ground}_{prompt_slug}_tool_rename_passk_comparison.png",
            y_max=1.0,
        )


def _aggregate_prompt_summary_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = ("pass@1", "pass@2", "pass@3", "accuracy")
    aggregated: Dict[str, Any] = {
        "files": "|".join(sorted({str(row.get("file") or "") for row in rows if str(row.get("file") or "").strip()})),
        "question_total": sum(_to_int(row.get("question_total")) for row in rows),
    }
    for metric in metrics:
        aggregated[metric] = _mean([_to_float(row.get(metric)) for row in rows])
    return aggregated


def _build_prompt_version_comparison_rows(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in summary_rows:
        prompt_version = _row_prompt_version(row)
        if prompt_version not in {"simple-prompt", "complex-prompt"}:
            continue
        key = (
            str(row.get("ground") or "unknown"),
            str(row.get("model") or ""),
        )
        grouped[key][prompt_version].append(row)

    comparison_rows: List[Dict[str, Any]] = []
    for (ground, model), by_prompt in grouped.items():
        simple_rows = by_prompt.get("simple-prompt", [])
        complex_rows = by_prompt.get("complex-prompt", [])
        if not simple_rows or not complex_rows:
            continue

        simple_metrics = _aggregate_prompt_summary_rows(simple_rows)
        complex_metrics = _aggregate_prompt_summary_rows(complex_rows)
        training_stage = _infer_training_stage(model, simple_metrics.get("files"), complex_metrics.get("files"))
        comparison_group = _comparison_group_for_stage(training_stage)

        def _delta(metric: str) -> float:
            return _to_float(complex_metrics.get(metric)) - _to_float(simple_metrics.get(metric))

        comparison_rows.append(
            {
                "ground": ground,
                "comparison_group": comparison_group,
                "comparison_group_label": _comparison_group_label(comparison_group),
                "model": model,
                "training_stage": training_stage,
                "simple_prompt_files": str(simple_metrics.get("files") or ""),
                "complex_prompt_files": str(complex_metrics.get("files") or ""),
                "question_total_simple": _to_int(simple_metrics.get("question_total")),
                "question_total_complex": _to_int(complex_metrics.get("question_total")),
                "pass@1_simple": _to_float(simple_metrics.get("pass@1")),
                "pass@1_complex": _to_float(complex_metrics.get("pass@1")),
                "delta_pass@1_complex_minus_simple": _delta("pass@1"),
                "pass@2_simple": _to_float(simple_metrics.get("pass@2")),
                "pass@2_complex": _to_float(complex_metrics.get("pass@2")),
                "delta_pass@2_complex_minus_simple": _delta("pass@2"),
                "pass@3_simple": _to_float(simple_metrics.get("pass@3")),
                "pass@3_complex": _to_float(complex_metrics.get("pass@3")),
                "delta_pass@3_complex_minus_simple": _delta("pass@3"),
                "accuracy_simple": _to_float(simple_metrics.get("accuracy")),
                "accuracy_complex": _to_float(complex_metrics.get("accuracy")),
                "delta_accuracy_complex_minus_simple": _delta("accuracy"),
            }
        )

    comparison_rows.sort(
        key=lambda row: (
            str(row.get("ground") or ""),
            str(row.get("comparison_group") or ""),
            _model_display_sort_key(row.get("model")),
            str(row.get("model") or ""),
        )
    )
    return comparison_rows


def _build_overall_model_metrics_prompt_split_rows(
    *,
    summary_rows: List[Dict[str, Any]],
    question_rows: List[Dict[str, Any]],
    trajectory_rows: List[Dict[str, Any]],
    counting_focus_summary_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    valid_prompts = {"simple-prompt", "complex-prompt"}
    grouped_summary: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    grouped_question: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    grouped_trajectory: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    grouped_counting: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    parent_keys: Set[Tuple[str, str]] = set()

    def _group_key(row: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
        prompt_version = _row_prompt_version(row)
        if prompt_version not in valid_prompts:
            return None
        ground = str(row.get("ground") or "unknown")
        model = str(row.get("model") or "")
        if not model:
            return None
        parent_keys.add((ground, model))
        return (ground, model, prompt_version)

    for row in summary_rows:
        key = _group_key(row)
        if key is not None:
            grouped_summary[key].append(row)
    for row in question_rows:
        key = _group_key(row)
        if key is not None:
            grouped_question[key].append(row)
    for row in trajectory_rows:
        key = _group_key(row)
        if key is not None:
            grouped_trajectory[key].append(row)
    for row in counting_focus_summary_rows:
        key = _group_key(row)
        if key is not None:
            grouped_counting[key].append(row)

    def _aggregate_scope(ground: str, model: str, prompt_version: str) -> Optional[Dict[str, Any]]:
        key = (ground, model, prompt_version)
        s_items = grouped_summary.get(key, [])
        q_items = grouped_question.get(key, [])
        t_items = grouped_trajectory.get(key, [])
        c_items = grouped_counting.get(key, [])
        if not s_items and not q_items and not t_items and not c_items:
            return None

        file_names = sorted(
            {
                str(item.get("file") or "").strip()
                for item in [*s_items, *q_items, *t_items, *c_items]
                if str(item.get("file") or "").strip()
            }
        )
        consumed_rows = [row for row in t_items if _to_int(row.get("consumed")) == 1]
        consumed_function_calls = [_to_float(row.get("function_calls")) for row in consumed_rows]

        question_total = len(q_items)
        reference_steps_total = sum(_to_float(row.get("reference_steps")) for row in q_items)
        tool_calls_consumed_total = sum(_to_float(row.get("consumed_tool_calls")) for row in q_items)
        illegal_function_call_errors_consumed = sum(
            _to_float(row.get("consumed_function_call_errors")) for row in q_items
        )
        no_function_call_trajectories_consumed = sum(
            1 for row in consumed_rows if _to_int(row.get("function_calls")) == 0
        )
        counting_question_total = sum(_to_int(row.get("counting_question_total")) for row in c_items)
        counting_trajectory_total = sum(_to_int(row.get("counting_trajectory_total")) for row in c_items)
        counting_tool_name_matched_steps = sum(_to_int(row.get("tool_name_matched_steps")) for row in c_items)

        return {
            "files": "|".join(file_names),
            "file_count": len(file_names),
            "question_total": question_total,
            "trajectory_total_consumed": len(consumed_rows),
            "accuracy": _mean([_to_float(row.get("accuracy")) for row in q_items]),
            "pass@1": _mean([_to_float(row.get("pass@1")) for row in q_items]),
            "pass@2": _mean([_to_float(row.get("pass@2")) for row in q_items]),
            "pass@3": _mean([_to_float(row.get("pass@3")) for row in q_items]),
            "avg_tries_used": _mean([_to_float(row.get("tries_used")) for row in q_items]),
            "reference_steps_total": reference_steps_total,
            "tool_calls_consumed_total": tool_calls_consumed_total,
            "efficiency_consumed": _safe_div(tool_calls_consumed_total, reference_steps_total),
            "avg_tool_calls_consumed_per_question": _safe_div(tool_calls_consumed_total, question_total),
            "illegal_function_call_errors_consumed": illegal_function_call_errors_consumed,
            "illegal_function_call_error_rate_consumed": _safe_div(
                illegal_function_call_errors_consumed,
                tool_calls_consumed_total,
            ),
            "tools_any_order_rate_consumed": _mean(
                [_to_float(row.get("tools_any_order")) for row in consumed_rows]
            ),
            "tools_in_order_rate_consumed": _mean(
                [_to_float(row.get("tools_in_order")) for row in consumed_rows]
            ),
            "tools_exact_match_rate_consumed": _mean(
                [_to_float(row.get("tools_exact_match")) for row in consumed_rows]
            ),
            "parameters_score_mean_consumed": _mean(
                [_to_float(row.get("parameters_score")) for row in consumed_rows]
            ),
            "semantic_judged_trajectories": sum(_to_int(row.get("judged")) for row in t_items),
            "semantic_cache_hits": sum(_to_int(row.get("semantic_cache_hit")) for row in t_items),
            "semantic_api_calls": sum(_to_int(row.get("semantic_api_called")) for row in t_items),
            "no_function_call_trajectories_consumed": no_function_call_trajectories_consumed,
            "no_function_call_rate_consumed": _safe_div(
                no_function_call_trajectories_consumed,
                len(consumed_rows),
            ),
            "missing_final_answer_marker_consumed": sum(
                _to_int(row.get("consumed_missing_final_answer_markers")) for row in q_items
            ),
            "missing_gt_questions": sum(_to_int(row.get("missing_gt_questions")) for row in s_items),
            "consumed_function_calls_mean": _mean(consumed_function_calls),
            "consumed_function_calls_median": _median(consumed_function_calls),
            "consumed_function_calls_max": max(consumed_function_calls) if consumed_function_calls else 0.0,
            "counting_question_total": counting_question_total,
            "counting_trajectory_total": counting_trajectory_total,
            "counting_right_tool_any_order_rate": _weighted_mean_from_rows(
                c_items,
                "right_tool_any_order_rate",
                "counting_trajectory_total",
            ),
            "counting_right_tool_in_order_rate": _weighted_mean_from_rows(
                c_items,
                "right_tool_in_order_rate",
                "counting_trajectory_total",
            ),
            "counting_right_tool_exact_match_rate": _weighted_mean_from_rows(
                c_items,
                "right_tool_exact_match_rate",
                "counting_trajectory_total",
            ),
            "counting_tool_name_matched_steps": counting_tool_name_matched_steps,
            "counting_args_exact_match_rate": _weighted_mean_from_rows(
                c_items,
                "args_exact_match_rate",
                "tool_name_matched_steps",
            ),
            "counting_args_semantic_same_rate": _weighted_mean_from_rows(
                c_items,
                "args_semantic_same_rate",
                "tool_name_matched_steps",
            ),
            "counting_args_semantic_same_steps": sum(
                _to_int(row.get("args_semantic_same_steps")) for row in c_items
            ),
            "counting_args_semantic_cache_hits": sum(
                _to_int(row.get("args_semantic_cache_hits")) for row in c_items
            ),
            "counting_args_semantic_api_calls": sum(
                _to_int(row.get("args_semantic_api_calls")) for row in c_items
            ),
        }

    aggregated_by_prompt: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for ground, model in parent_keys:
        for prompt_version in ("simple-prompt", "complex-prompt"):
            aggregated = _aggregate_scope(ground, model, prompt_version)
            if aggregated is not None:
                aggregated_by_prompt[(ground, model)][prompt_version] = aggregated

    rows: List[Dict[str, Any]] = []
    for (ground, model), by_prompt in sorted(
        aggregated_by_prompt.items(),
        key=lambda item: (
            item[0][0],
            _comparison_group_for_stage(_infer_training_stage(item[0][1])),
            _model_display_sort_key(item[0][1]),
            item[0][1],
        ),
    ):
        simple = by_prompt.get("simple-prompt")
        complex_ = by_prompt.get("complex-prompt")
        training_stage = _infer_training_stage(
            model,
            (simple or {}).get("files"),
            (complex_ or {}).get("files"),
        )
        comparison_group = _comparison_group_for_stage(training_stage)

        row: Dict[str, Any] = {
            "ground": ground,
            "comparison_group": comparison_group,
            "comparison_group_label": _comparison_group_label(comparison_group),
            "model": model,
            "training_stage": training_stage,
            "has_simple_prompt": int(simple is not None),
            "has_complex_prompt": int(complex_ is not None),
            "simple_prompt_files": (simple or {}).get("files", ""),
            "complex_prompt_files": (complex_ or {}).get("files", ""),
            "simple_prompt_file_count": (simple or {}).get("file_count", 0),
            "complex_prompt_file_count": (complex_ or {}).get("file_count", 0),
        }

        for metric in OVERALL_MODEL_METRICS_PROMPT_SPLIT_METRICS:
            row[f"{metric}_simple"] = "" if simple is None else simple.get(metric, "")
            row[f"{metric}_complex"] = "" if complex_ is None else complex_.get(metric, "")

        def _delta(metric: str) -> Any:
            if simple is None or complex_ is None:
                return ""
            return _to_float(complex_.get(metric)) - _to_float(simple.get(metric))

        row["delta_accuracy_complex_minus_simple"] = _delta("accuracy")
        row["delta_pass@1_complex_minus_simple"] = _delta("pass@1")
        row["delta_pass@2_complex_minus_simple"] = _delta("pass@2")
        row["delta_pass@3_complex_minus_simple"] = _delta("pass@3")
        row["delta_avg_tries_used_complex_minus_simple"] = _delta("avg_tries_used")
        row["delta_efficiency_consumed_complex_minus_simple"] = _delta("efficiency_consumed")
        row["delta_avg_tool_calls_consumed_per_question_complex_minus_simple"] = _delta(
            "avg_tool_calls_consumed_per_question"
        )
        row["delta_illegal_function_call_error_rate_consumed_complex_minus_simple"] = _delta(
            "illegal_function_call_error_rate_consumed"
        )
        row["delta_no_function_call_rate_consumed_complex_minus_simple"] = _delta(
            "no_function_call_rate_consumed"
        )
        row["delta_consumed_function_calls_mean_complex_minus_simple"] = _delta(
            "consumed_function_calls_mean"
        )
        row["delta_counting_args_semantic_same_rate_complex_minus_simple"] = _delta(
            "counting_args_semantic_same_rate"
        )
        rows.append(row)

    return rows


def _generate_prompt_version_comparison_charts(*, chart_dir: Path, comparison_rows: List[Dict[str, Any]]) -> None:
    if not comparison_rows:
        return
    plt = _load_matplotlib()
    if plt is None:
        return

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        key = (
            str(row.get("ground") or "unknown"),
            str(row.get("comparison_group") or "base"),
        )
        grouped[key].append(row)

    chart_dir.mkdir(parents=True, exist_ok=True)
    for (ground, comparison_group), rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda row: (_model_display_sort_key(row.get("model")), str(row.get("model") or "")))
        labels = [str(row.get("model") or "") for row in rows_sorted]
        group_label = _comparison_group_label(comparison_group)

        _plot_grouped_bars(
            plt,
            title=f"{group_label}: Accuracy by Prompt ({ground})",
            labels=labels,
            series=[
                ("Simple Prompt", [_to_float(row.get("accuracy_simple")) for row in rows_sorted]),
                ("Complex Prompt", [_to_float(row.get("accuracy_complex")) for row in rows_sorted]),
            ],
            ylabel="Accuracy",
            output_path=chart_dir / f"{ground}_{comparison_group}_prompt_accuracy_comparison.png",
            y_max=1.0,
        )

        delta_values = [
            _to_float(row.get("delta_pass@1_complex_minus_simple")) for row in rows_sorted
        ] + [
            _to_float(row.get("delta_pass@2_complex_minus_simple")) for row in rows_sorted
        ] + [
            _to_float(row.get("delta_pass@3_complex_minus_simple")) for row in rows_sorted
        ]
        y_min = min([-0.05] + delta_values) if delta_values else -0.05
        y_max = max([0.05] + delta_values) if delta_values else 0.05
        _plot_grouped_bars(
            plt,
            title=f"{group_label}: Prompt Delta (Complex - Simple) ({ground})",
            labels=labels,
            series=[
                ("Pass@1 Delta", [_to_float(row.get("delta_pass@1_complex_minus_simple")) for row in rows_sorted]),
                ("Pass@2 Delta", [_to_float(row.get("delta_pass@2_complex_minus_simple")) for row in rows_sorted]),
                ("Pass@3 Delta", [_to_float(row.get("delta_pass@3_complex_minus_simple")) for row in rows_sorted]),
            ],
            ylabel="Delta Rate",
            output_path=chart_dir / f"{ground}_{comparison_group}_prompt_passk_delta.png",
            y_min=y_min,
            y_max=y_max,
            draw_zero_line=True,
        )


def _generate_combined_charts(
    *,
    chart_dir: Path,
    summary_rows: List[Dict[str, Any]],
    title_suffix: str = "",
    file_suffix: str = "",
) -> None:
    plt = _load_matplotlib()
    if plt is None:
        return
    means = _build_ground_means(summary_rows)
    if not means:
        return

    grounds = sorted(means.keys())

    def _vals(metric: str) -> List[float]:
        return [means[g].get(metric, 0.0) for g in grounds]

    _plot_grouped_bars(
        plt,
        title=f"Ground Comparison: Pass@k{title_suffix}",
        labels=grounds,
        series=[
            ("Pass@1", _vals("pass@1")),
            ("Pass@2", _vals("pass@2")),
            ("Pass@3", _vals("pass@3")),
        ],
        ylabel="Rate",
        output_path=chart_dir / f"accuracy_passk_comparison{file_suffix}.png",
        y_max=1.0,
    )

    _plot_grouped_bars(
        plt,
        title=f"Ground Comparison: Efficiency{title_suffix}",
        labels=grounds,
        series=[
            ("Efficiency", _vals("efficiency_consumed")),
        ],
        ylabel="Efficiency",
        output_path=chart_dir / f"efficiency_comparison{file_suffix}.png",
    )

    _plot_grouped_bars(
        plt,
        title=f"Ground Comparison: Illegal Function-Call Error Rate{title_suffix}",
        labels=grounds,
        series=[
            ("Error Rate", _vals("illegal_function_call_error_rate_consumed")),
        ],
        ylabel="Error Rate",
        output_path=chart_dir / f"illegal_error_rate_comparison{file_suffix}.png",
        y_max=1.0,
    )

    _plot_grouped_bars(
        plt,
        title=f"Ground Comparison: Tool-Step Rates (Consumed Scope){title_suffix}",
        labels=grounds,
        series=[
            ("Any Order", _vals("tools_any_order_rate_consumed")),
            ("In Order", _vals("tools_in_order_rate_consumed")),
            ("Exact Match", _vals("tools_exact_match_rate_consumed")),
            ("Param Mean", _vals("parameters_score_mean_consumed")),
        ],
        ylabel="Rate",
        output_path=chart_dir / f"tool_step_rate_comparison{file_suffix}.png",
        y_max=1.0,
    )


def _write_tool_rename_comparison_outputs(
    *,
    variant_output_root: Path,
    summary_rows: List[Dict[str, Any]],
    no_charts: bool,
) -> None:
    comparison_rows = _build_tool_rename_comparison_rows(summary_rows)
    comparison_dir = variant_output_root / "tool_rename_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        comparison_dir / "tool_rename_4b_comparison.csv",
        TOOL_RENAME_COMPARISON_FIELDNAMES,
        comparison_rows,
    )
    if not no_charts and comparison_rows:
        _generate_tool_rename_comparison_charts(
            chart_dir=comparison_dir / "charts",
            comparison_rows=comparison_rows,
        )
    if comparison_rows:
        print(
            f"[INFO] Wrote tool-rename comparison to "
            f"{comparison_dir / 'tool_rename_4b_comparison.csv'}"
        )
    else:
        print("[INFO] No tool-rename 4B rows found; comparison CSV contains header only.")


def _write_prompt_version_comparison_outputs(
    *,
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    no_charts: bool,
) -> None:
    comparison_rows = _build_prompt_version_comparison_rows(summary_rows)
    comparison_dir = output_dir / "prompt_version_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        comparison_dir / "prompt_version_comparison.csv",
        PROMPT_VERSION_COMPARISON_FIELDNAMES,
        comparison_rows,
    )
    if not no_charts and comparison_rows:
        _generate_prompt_version_comparison_charts(
            chart_dir=comparison_dir / "charts",
            comparison_rows=comparison_rows,
        )
    if comparison_rows:
        print(
            f"[INFO] Wrote prompt-version comparison to "
            f"{comparison_dir / 'prompt_version_comparison.csv'}"
        )
    else:
        print("[INFO] No simple/complex prompt pairs found; prompt comparison CSV contains header only.")


def _write_overall_model_metrics_prompt_split_output(
    *,
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    question_rows: List[Dict[str, Any]],
    trajectory_rows: List[Dict[str, Any]],
    counting_focus_summary_rows: List[Dict[str, Any]],
) -> None:
    rows = _build_overall_model_metrics_prompt_split_rows(
        summary_rows=summary_rows,
        question_rows=question_rows,
        trajectory_rows=trajectory_rows,
        counting_focus_summary_rows=counting_focus_summary_rows,
    )
    output_path = output_dir / "overall_model_metrics_prompt_split.csv"
    _write_csv(
        output_path,
        OVERALL_MODEL_METRICS_PROMPT_SPLIT_FIELDNAMES,
        rows,
    )
    if rows:
        print(f"[INFO] Wrote overall prompt-split metrics to {output_path}")
    else:
        print("[INFO] No simple/complex prompt rows found; overall prompt-split CSV contains header only.")


def _run_ground_pipeline(
    *,
    ground: str,
    evaluations: List[FileEvaluation],
    output_dir: Path,
    no_charts: bool,
    parameter_semantic_matcher: ParameterSemanticMatcher,
    semantic_workers: int,
    show_progress: bool,
) -> Dict[str, Any]:
    overall_evaluations = [
        eval_result
        for eval_result in evaluations
        if not _is_tool_rename_artifact(model=eval_result.model, file_name=eval_result.file)
    ]
    excluded_tool_rename_files = len(evaluations) - len(overall_evaluations)

    question_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    for eval_result in overall_evaluations:
        question_rows.extend(eval_result.question_rows)
        trajectory_rows.extend(eval_result.trajectory_rows)
        step_rows.extend(eval_result.step_rows)

    summary_rows = _build_summary_rows(evaluations=overall_evaluations)
    summary_rows_all = _build_summary_rows(evaluations=evaluations)
    traj_len_rows = _build_trajectory_length_distribution(trajectory_rows)
    tries_rows = _build_tries_used_distribution(question_rows)
    function_call_error_rows = _build_function_call_error_summary(question_rows)
    no_function_call_rows = _build_no_function_call_summary(trajectory_rows)
    tool_step_rows = _build_tool_step_summary(trajectory_rows)
    question_type_rows = _build_question_type_breakdown(question_rows, trajectory_rows)
    counting_focus_summary_rows, counting_focus_step_rows = _build_counting_focus_analysis(
        question_rows=question_rows,
        trajectory_rows=trajectory_rows,
        step_rows=step_rows,
        parameter_semantic_matcher=parameter_semantic_matcher,
        semantic_workers=semantic_workers,
        show_progress=show_progress,
    )

    display_prompt_version = _select_reference_prompt_version(summary_rows)
    display_summary_rows = list(summary_rows)
    display_question_rows = list(question_rows)
    display_trajectory_rows = list(trajectory_rows)
    display_counting_focus_rows = list(counting_focus_summary_rows)
    if display_prompt_version is not None:
        filtered_summary_rows = _filter_rows_by_prompt_version(summary_rows, display_prompt_version)
        filtered_question_rows = _filter_rows_by_prompt_version(question_rows, display_prompt_version)
        filtered_trajectory_rows = _filter_rows_by_prompt_version(trajectory_rows, display_prompt_version)
        filtered_counting_focus_rows = _filter_rows_by_prompt_version(
            counting_focus_summary_rows,
            display_prompt_version,
        )
        if filtered_summary_rows:
            display_summary_rows = filtered_summary_rows
            display_question_rows = filtered_question_rows
            display_trajectory_rows = filtered_trajectory_rows
            display_counting_focus_rows = filtered_counting_focus_rows

    prompt_title_suffix = ""
    if display_prompt_version is not None and len(display_summary_rows) != len(summary_rows):
        prompt_title_suffix = f" ({_prompt_version_display(display_prompt_version)})"

    _write_csv(output_dir / "summary_by_model.csv", SUMMARY_FIELDNAMES, summary_rows)
    _write_csv(output_dir / "question_outcomes.csv", QUESTION_OUTCOME_FIELDNAMES, question_rows)
    _write_csv(output_dir / "trajectory_metrics.csv", TRAJECTORY_METRICS_FIELDNAMES, trajectory_rows)
    _write_csv(output_dir / "step_details.csv", STEP_DETAILS_FIELDNAMES, step_rows)
    _write_csv(output_dir / "trajectory_length_distribution.csv", TRAJ_LENGTH_FIELDNAMES, traj_len_rows)
    _write_csv(output_dir / "tries_used_distribution.csv", TRIES_USED_FIELDNAMES, tries_rows)
    _write_csv(output_dir / "function_call_error_summary.csv", FUNCTION_CALL_ERROR_FIELDNAMES, function_call_error_rows)
    _write_csv(output_dir / "no_function_call_summary.csv", NO_FUNCTION_CALL_FIELDNAMES, no_function_call_rows)
    _write_csv(output_dir / "tool_step_summary.csv", TOOL_STEP_SUMMARY_FIELDNAMES, tool_step_rows)
    _write_csv(output_dir / "question_type_breakdown.csv", QUESTION_TYPE_BREAKDOWN_FIELDNAMES, question_type_rows)
    _write_csv(
        output_dir / "counting_tool_parameter_focus_summary.csv",
        COUNTING_FOCUS_SUMMARY_FIELDNAMES,
        counting_focus_summary_rows,
    )
    _write_csv(
        output_dir / "counting_tool_parameter_focus_step_details.csv",
        COUNTING_FOCUS_STEP_FIELDNAMES,
        counting_focus_step_rows,
    )
    _write_prompt_version_comparison_outputs(
        output_dir=output_dir,
        summary_rows=summary_rows,
        no_charts=no_charts,
    )
    _write_overall_model_metrics_prompt_split_output(
        output_dir=output_dir,
        summary_rows=summary_rows,
        question_rows=question_rows,
        trajectory_rows=trajectory_rows,
        counting_focus_summary_rows=counting_focus_summary_rows,
    )

    if not no_charts:
        _generate_ground_charts(
            output_dir=output_dir,
            summary_rows=display_summary_rows,
            question_rows=display_question_rows,
            trajectory_rows=display_trajectory_rows,
            title_suffix=prompt_title_suffix,
        )
        for prompt_version in _available_prompt_versions(summary_rows):
            prompt_summary_rows = _filter_rows_by_prompt_version(summary_rows, prompt_version)
            if not prompt_summary_rows:
                continue
            _generate_ground_charts(
                output_dir=output_dir,
                summary_rows=prompt_summary_rows,
                question_rows=_filter_rows_by_prompt_version(question_rows, prompt_version),
                trajectory_rows=_filter_rows_by_prompt_version(trajectory_rows, prompt_version),
                title_suffix=f" ({_prompt_version_display(prompt_version)})",
                file_suffix=f"_{prompt_version.replace('-', '_')}",
            )
        _generate_counting_focus_charts(
            output_dir=output_dir,
            counting_focus_summary_rows=display_counting_focus_rows,
        )

    print(
        f"[INFO] Ground={ground}: files={len(overall_evaluations)}, "
        f"excluded_tool_rename_files={excluded_tool_rename_files}, questions={len(question_rows)}, "
        f"trajectories={len(trajectory_rows)}, display_prompt_version={display_prompt_version or 'all'}"
    )

    return {
        "summary_rows": summary_rows,
        "summary_rows_all": summary_rows_all,
        "question_rows": question_rows,
        "trajectory_rows": trajectory_rows,
        "step_rows": step_rows,
        "traj_len_rows": traj_len_rows,
        "tries_rows": tries_rows,
        "function_call_error_rows": function_call_error_rows,
        "no_function_call_rows": no_function_call_rows,
        "tool_step_rows": tool_step_rows,
        "question_type_rows": question_type_rows,
        "counting_focus_summary_rows": counting_focus_summary_rows,
        "counting_focus_step_rows": counting_focus_step_rows,
    }


def parse_args() -> argparse.Namespace:
    default_input_dir = PROJECT_ROOT / "Agent_TEST_outputs"
    default_gt_jsonl = PROJECT_ROOT / "eo_gym_test_datasets" / "eo_gym_test_full.jsonl"
    default_question_type_csv = PROJECT_ROOT / "EO_GYM_datasets" / "eo_gym_test_dataset_traj.csv"

    parser = argparse.ArgumentParser(
        description="Evaluate open-text agent outputs with early-stop pass@k and semantic matching."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Directory containing model output JSONL files (default: Agent_TEST_outputs).",
    )
    parser.add_argument(
        "--gt-jsonl",
        type=Path,
        default=default_gt_jsonl,
        help="Ground truth JSONL path (default: eo_gym_test_full.jsonl).",
    )
    parser.add_argument(
        "--question-type-csv",
        type=Path,
        default=default_question_type_csv,
        help="CSV containing uid->question_type mapping (default: EO_GYM_datasets/eo_gym_test_dataset_traj.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root folder (default: <input-dir>/agent_test_eval_passk).",
    )
    parser.add_argument(
        "--max-traj",
        type=int,
        default=3,
        help="Maximum trajectory index count to evaluate per question (default: 3).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4.1-mini",
        help="Semantic judge model (default: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--semantic-cache",
        type=Path,
        default=None,
        help="Semantic matching cache JSONL path (default under output root cache/).",
    )
    parser.add_argument(
        "--parameter-semantic-cache",
        type=Path,
        default=None,
        help="Tool-argument semantic cache JSONL path (default under output root cache/).",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional debug limit: evaluate at most N questions per file.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="*.jsonl",
        help="Glob pattern for input files inside --input-dir (default: *.jsonl).",
    )
    parser.add_argument(
        "--semantic-workers",
        type=int,
        default=8,
        help="Thread workers for semantic matching at question level (default: 8).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--skip-zero-function-call-filtered-version",
        action="store_true",
        help=(
            "Skip generating the additional result version that excludes trajectories with zero function calls."
        ),
    )

    args = parser.parse_args()
    if args.max_traj <= 0:
        raise ValueError("--max-traj must be >= 1")
    if args.semantic_workers <= 0:
        raise ValueError("--semantic-workers must be >= 1")
    return args


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    gt_jsonl: Path = args.gt_jsonl
    question_type_csv: Path = args.question_type_csv

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not gt_jsonl.exists():
        raise FileNotFoundError(f"GT JSONL not found: {gt_jsonl}")
    if not question_type_csv.exists():
        raise FileNotFoundError(f"Question-type CSV not found: {question_type_csv}")

    output_root = args.output_root or (input_dir / "agent_test_eval_passk")
    output_root.mkdir(parents=True, exist_ok=True)

    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.judge_model)
    semantic_cache = args.semantic_cache
    if semantic_cache is None:
        semantic_cache = output_root / "cache" / f"semantic_match_{safe_model}.jsonl"
    parameter_semantic_cache = args.parameter_semantic_cache
    if parameter_semantic_cache is None:
        parameter_semantic_cache = output_root / "cache" / f"parameter_semantic_match_{safe_model}.jsonl"

    print(f"[INFO] Loading GT from {gt_jsonl}")
    gt_map = _load_ground_truth(gt_jsonl)
    print(f"[INFO] Loaded {len(gt_map)} GT questions")
    print(f"[INFO] Loading question-type mapping from {question_type_csv}")
    uid_question_type_map = _load_question_type_map(question_type_csv)
    print(f"[INFO] Loaded {len(uid_question_type_map)} uid->question_type mappings")

    matcher = SemanticMatcher(
        cache_path=semantic_cache,
        judge_model=args.judge_model,
        prompt_version=JUDGE_PROMPT_VERSION,
    )
    parameter_matcher = ParameterSemanticMatcher(
        cache_path=parameter_semantic_cache,
        judge_model=args.judge_model,
        prompt_version=PARAMETER_JUDGE_PROMPT_VERSION,
    )

    try:
        jsonl_files = sorted(path for path in input_dir.glob(args.input_glob) if path.is_file())
        if not jsonl_files:
            print(f"[WARN] No files matched {args.input_glob} in {input_dir}")
            return

        def _run_variant(
            *,
            variant_label: str,
            variant_output_root: Path,
            min_function_calls: int,
            include_empty_questions: bool,
        ) -> None:
            include_cross_model_analysis = True
            evaluations_by_ground: Dict[str, List[FileEvaluation]] = defaultdict(list)

            file_iter: Iterable[Path] = jsonl_files
            if tqdm is not None:
                file_iter = tqdm(
                    jsonl_files,
                    total=len(jsonl_files),
                    desc=f"Evaluating files ({variant_label})",
                    leave=True,
                    disable=args.no_progress,
                )

            for path in file_iter:
                eval_result = _evaluate_prediction_file(
                    path=path,
                    gt_map=gt_map,
                    uid_question_type_map=uid_question_type_map,
                    semantic_matcher=matcher,
                    max_traj=args.max_traj,
                    max_questions=args.max_questions,
                    semantic_workers=args.semantic_workers,
                    show_progress=not args.no_progress,
                    min_function_calls=min_function_calls,
                    include_empty_questions=include_empty_questions,
                )
                if eval_result is None:
                    continue
                evaluations_by_ground[eval_result.ground].append(eval_result)
                print(
                    f"[INFO] [{variant_label}] Evaluated {path.name} -> ground={eval_result.ground}, "
                    f"questions={len(eval_result.question_rows)}, trajectories={len(eval_result.trajectory_rows)}"
                )

            if not evaluations_by_ground:
                print(f"[WARN] [{variant_label}] No evaluation records found for configured grounds.")
                return

            ground_results: List[Dict[str, Any]] = []
            for ground in ("all-tools", "mapped-build-in-functions"):
                evals = evaluations_by_ground.get(ground, [])
                if not evals:
                    continue
                ground_output_dir = variant_output_root / ground
                result = _run_ground_pipeline(
                    ground=ground,
                    evaluations=evals,
                    output_dir=ground_output_dir,
                    no_charts=args.no_charts,
                    parameter_semantic_matcher=parameter_matcher,
                    semantic_workers=args.semantic_workers,
                    show_progress=not args.no_progress,
                )
                ground_results.append(result)
                print(f"[INFO] [{variant_label}] Wrote ground outputs to {ground_output_dir}")

            if include_cross_model_analysis:
                all_summary_rows: List[Dict[str, Any]] = []
                all_summary_rows_for_tool_rename: List[Dict[str, Any]] = []
                all_question_rows: List[Dict[str, Any]] = []
                all_trajectory_rows: List[Dict[str, Any]] = []
                all_question_type_rows: List[Dict[str, Any]] = []
                all_no_function_call_rows: List[Dict[str, Any]] = []
                all_counting_focus_rows: List[Dict[str, Any]] = []

                for result in ground_results:
                    all_summary_rows.extend(result["summary_rows"])
                    all_summary_rows_for_tool_rename.extend(result["summary_rows_all"])
                    all_question_rows.extend(result["question_rows"])
                    all_trajectory_rows.extend(result["trajectory_rows"])
                    all_question_type_rows.extend(result["question_type_rows"])
                    all_no_function_call_rows.extend(result["no_function_call_rows"])
                    all_counting_focus_rows.extend(result["counting_focus_summary_rows"])

                combined_dir = variant_output_root / "combined"
                combined_chart_dir = combined_dir / "ground_comparison_charts"
                combined_dir.mkdir(parents=True, exist_ok=True)

                _write_csv(
                    combined_dir / "ground_comparison_summary.csv",
                    SUMMARY_FIELDNAMES,
                    all_summary_rows,
                )
                _write_csv(
                    combined_dir / "question_type_breakdown.csv",
                    QUESTION_TYPE_BREAKDOWN_FIELDNAMES,
                    all_question_type_rows,
                )
                _write_csv(
                    combined_dir / "no_function_call_summary.csv",
                    NO_FUNCTION_CALL_FIELDNAMES,
                    all_no_function_call_rows,
                )
                _write_csv(
                    combined_dir / "counting_tool_parameter_focus_summary.csv",
                    COUNTING_FOCUS_SUMMARY_FIELDNAMES,
                    all_counting_focus_rows,
                )
                _write_prompt_version_comparison_outputs(
                    output_dir=combined_dir,
                    summary_rows=all_summary_rows,
                    no_charts=args.no_charts,
                )
                _write_overall_model_metrics_prompt_split_output(
                    output_dir=combined_dir,
                    summary_rows=all_summary_rows,
                    question_rows=all_question_rows,
                    trajectory_rows=all_trajectory_rows,
                    counting_focus_summary_rows=all_counting_focus_rows,
                )

                if not args.no_charts:
                    combined_display_prompt_version = _select_reference_prompt_version(all_summary_rows)
                    combined_display_rows = list(all_summary_rows)
                    combined_title_suffix = ""
                    if combined_display_prompt_version is not None:
                        filtered_combined_rows = _filter_rows_by_prompt_version(
                            all_summary_rows,
                            combined_display_prompt_version,
                        )
                        if filtered_combined_rows:
                            combined_display_rows = filtered_combined_rows
                        if len(combined_display_rows) != len(all_summary_rows):
                            combined_title_suffix = (
                                f" ({_prompt_version_display(combined_display_prompt_version)})"
                            )
                    _generate_combined_charts(
                        chart_dir=combined_chart_dir,
                        summary_rows=combined_display_rows,
                        title_suffix=combined_title_suffix,
                    )
                    for prompt_version in _available_prompt_versions(all_summary_rows):
                        prompt_summary_rows = _filter_rows_by_prompt_version(all_summary_rows, prompt_version)
                        if not prompt_summary_rows:
                            continue
                        _generate_combined_charts(
                            chart_dir=combined_chart_dir,
                            summary_rows=prompt_summary_rows,
                            title_suffix=f" ({_prompt_version_display(prompt_version)})",
                            file_suffix=f"_{prompt_version.replace('-', '_')}",
                        )

                _write_tool_rename_comparison_outputs(
                    variant_output_root=variant_output_root,
                    summary_rows=all_summary_rows_for_tool_rename,
                    no_charts=args.no_charts,
                )

                print(
                    f"[INFO] [{variant_label}] Wrote combined summary to "
                    f"{combined_dir / 'ground_comparison_summary.csv'}"
                )
            else:
                print(f"[INFO] [{variant_label}] Skipping cross-model analysis outputs.")

        _run_variant(
            variant_label="base",
            variant_output_root=output_root,
            min_function_calls=0,
            include_empty_questions=False,
        )

        if not args.skip_zero_function_call_filtered_version:
            filtered_output_root = output_root / "filtered_nonzero_function_calls"
            _run_variant(
                variant_label="filtered-no-empty-tools",
                variant_output_root=filtered_output_root,
                min_function_calls=1,
                include_empty_questions=True,
            )
            print(f"[INFO] Wrote filtered outputs to {filtered_output_root}")

        print(f"[INFO] Semantic cache path: {semantic_cache}")
        print(f"[INFO] Parameter semantic cache path: {parameter_semantic_cache}")

    finally:
        matcher.close()
        parameter_matcher.close()


if __name__ == "__main__":
    main()
