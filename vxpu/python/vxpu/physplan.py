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

"""Physical execution plans, per-site strategies, and execution knobs (#529).

This module defines the mapping from a physical plan's per-site strategy
choices to the engine's effective knobs.
"""

import hashlib
import json
import os

PLAN_FORMAT = "vxpu-plan/1"
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
