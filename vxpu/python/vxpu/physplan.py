# Copyright 2026 Google LLC
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

"""Physical execution plans, per-site strategies, and run records (#529).

This module defines the mapping from a physical plan's per-site strategy
choices to the engine's effective knobs, and the run record format written
at the end of an execution.
"""

import hashlib
import json
import os
import time

PLAN_FORMAT = "vxpu-plan/1"
RECORD_FORMAT = "vxpu-run-record/1"
RECORD_SCHEMA_VERSION = 1
BUILD_COMMIT = ""

KNOWN_STRATEGIES = {
    "dense GEMV": {"default"},
    "expert op": {"default", "fused", "per-expert"},
    "sum site": {"default"},
    "attention": {"default"},
    "drafter": {"default"},
}


def load_plan(plan):
    """Load a Plan dictionary from a dictionary, JSON string, or file path."""
    if isinstance(plan, dict):
        return plan
    if isinstance(plan, (str, bytes, os.PathLike)):
        if isinstance(plan, bytes):
            plan = plan.decode("utf-8")
        s = str(plan).strip()
        if s.startswith("{"):
            return json.loads(s)
        if not os.path.exists(s):
            raise FileNotFoundError(f"plan file not found: {s}")
        with open(s, "r", encoding="utf-8") as f:
            return json.load(f)
    raise TypeError(f"expected dict, path or JSON string for plan, got {type(plan)!r}")


def plan_hash(plan):
    """Compute the canonical 16-hex-digit SHA-256 hash of a Plan."""
    data = load_plan(plan)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


def compute_artifact_md5(manifest_bytes, binding_bytes):
    """Compute the MD5 checksum over an artifact's manifest and binding."""
    h = hashlib.md5()
    h.update(manifest_bytes)
    h.update(binding_bytes)
    return h.hexdigest()


def get_tree_commit():
    """Retrieve the current tree commit from the build or environment."""
    if BUILD_COMMIT:
        return BUILD_COMMIT
    for key in ("GIT_COMMIT", "TREE_COMMIT"):
        if os.environ.get(key):
            return os.environ[key]
    return "unknown"


def apply_env(plan, env=None):
    """Read the per-site strategy choices from a plan and return effective knobs.

    Validates strategies per class against KNOWN_STRATEGIES. Unknown
    strategies raise a ValueError. If env is provided, effective knobs are
    updated into env. Returns the dictionary of effective knobs.
    """
    plan_dict = load_plan(plan)

    knobs = {}
    for site in plan_dict.get("sites", []):
        site_id = site.get("id", "")
        site_class = site.get("class")
        strat = site.get("strategy", "default")
        if site_class not in KNOWN_STRATEGIES:
            raise ValueError(f"unknown site class {site_class!r} for site {site_id!r}")
        if strat not in KNOWN_STRATEGIES[site_class]:
            raise ValueError(f"unknown strategy {strat!r} for site {site_id!r} of class {site_class!r}")

        if site_class == "expert op":
            if strat == "fused":
                knobs["VXPU_FUSED_EXPERTS"] = "1"
            elif strat == "per-expert":
                knobs["VXPU_FUSED_EXPERTS"] = "0"

    if "switches" in plan_dict:
        for k, v in plan_dict["switches"].items():
            knobs[str(k)] = str(v)

    if env is not None:
        env.update(knobs)
    return knobs


def create_run_record(plan=None, knobs_in_effect=None, median_step_ms=0.0,
                      tokens_per_step=1.0, tokens_a_step=None, checks=None,
                      per_card_peak_memory=None, build_time_s=0.0, build_time=None,
                      per_class_times=None, plan_hash_val=None,
                      tree_commit=None, artifact_md5=None):
    """Build a versioned run record dictionary."""
    if plan is not None:
        plan_dict = load_plan(plan)
        if plan_hash_val is None:
            plan_hash_val = plan_hash(plan_dict)
        if tree_commit is None:
            tree_commit = plan_dict.get("tree_commit")
        if artifact_md5 is None:
            artifact_md5 = plan_dict.get("artifact_md5")

    if plan_hash_val is None:
        plan_hash_val = "unknown"
    if tree_commit is None:
        tree_commit = get_tree_commit()
    if artifact_md5 is None:
        artifact_md5 = "unknown"
    if knobs_in_effect is None:
        knobs_in_effect = {}
    if checks is None:
        checks = {"status": "not run"}
    if per_card_peak_memory is None:
        per_card_peak_memory = {}

    eff_tokens_per_step = tokens_per_step if tokens_a_step is None else tokens_a_step
    eff_build_time_s = build_time_s if build_time is None else build_time

    return {
        "format": RECORD_FORMAT,
        "schema_version": RECORD_SCHEMA_VERSION,
        "plan_hash": str(plan_hash_val),
        "tree_commit": str(tree_commit),
        "artifact_md5": str(artifact_md5),
        "knobs_in_effect": {str(k): str(v) for k, v in knobs_in_effect.items()},
        "median_step_ms": round(float(median_step_ms), 3),
        "tokens_per_step": float(eff_tokens_per_step),
        "checks": dict(checks),
        "per_card_peak_memory": dict(per_card_peak_memory),
        "build_time_s": round(float(eff_build_time_s), 3),
        "per_class_times": dict(per_class_times) if per_class_times is not None else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def parse_record(record_or_path):
    """Validate and parse a run record, checking schema version and required fields."""
    if isinstance(record_or_path, (str, bytes, os.PathLike)):
        if isinstance(record_or_path, bytes):
            record_or_path = record_or_path.decode("utf-8")
        s = str(record_or_path).strip()
        if s.startswith("{"):
            record = json.loads(s)
        else:
            if not os.path.exists(s):
                raise FileNotFoundError(f"run record file not found: {s}")
            with open(s, "r", encoding="utf-8") as f:
                record = json.load(f)
    elif isinstance(record_or_path, dict):
        record = dict(record_or_path)
    else:
        raise TypeError(f"expected dict, path or JSON string for record, got {type(record_or_path)!r}")

    if record.get("format") != RECORD_FORMAT:
        raise ValueError(f"unexpected record format {record.get('format')!r}, expected {RECORD_FORMAT!r}")
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise ValueError(f"unexpected record schema version {record.get('schema_version')!r}, expected {RECORD_SCHEMA_VERSION!r}")
    if not record.get("plan_hash"):
        raise ValueError("missing plan_hash")
    if not record.get("tree_commit"):
        raise ValueError("missing tree_commit")
    if not record.get("artifact_md5"):
        raise ValueError("missing artifact_md5")
    if "knobs_in_effect" not in record or record["knobs_in_effect"] is None:
        raise ValueError("missing knobs_in_effect")
    return record


def write_run_record(record, out_path):
    """Validate and write a run record to out_path as formatted JSON."""
    validated = parse_record(record)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validated, f, indent=2)
        f.write("\n")
