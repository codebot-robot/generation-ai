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

"""The executor's physical plan: what it chose to run, and why.

The artifact is the logical plan (the model's computation, #316); how to
run it on a given machine is the executor's choice, as a database picks a
physical plan for a query. A Plan is built once per engine from the
hardware and the configuration, and it is the one place those choices are
made and recorded:

  hardware     what the engine runs on (device, GPU, CPU, SIMD, RAM)
  choices      global decisions (split decode, compile mode, where the
               experts live, the bf16 weight cache, ...), each with its
               value, where the value came from (default, environment or
               hint) and the alternatives
  ops          for each logical op the graphs call (vxpu::fp8_linear,
               vxpu::moe_experts, ...), the registered implementations,
               when each applies, and its numerics class: bit-identical to
               the reference computation, or equal to rounding
  selections   which implementation each call site actually got, keyed by
               op, implementation and input shapes

Ops dispatch through Plan.select instead of deciding inline, so what runs
is what the plan says and what explain() reports. Selections are recorded
when Python dispatches an op: eager calls, tracing and CUDA-graph
recording. A replayed CUDA graph does not call Python, so the record says
which implementation a shape got, not how many times it ran.

`python -m vxpu.plan --artifact DIR [--device D]` explains the plan for an
artifact without loading its weights.
"""

import argparse
import collections
import dataclasses
import json
import os
import platform
import threading

# ---------------------------------------------------------------------------
# configuration: one parser for every switch (#320)

TRUE = ("1", "true", "on", "yes")
FALSE = ("0", "false", "off", "no")


class ConfigError(ValueError):
    """A switch with a value the parser does not understand."""


def env_flag(name, default, extra_false=()):
    """A boolean switch: 1/true/on/yes or 0/false/off/no, any case;
    anything else is an error rather than a guess. Returns (value, source)."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default, "default"
    v = raw.strip().lower()
    if v in TRUE:
        return True, "env"
    if v in FALSE or v in extra_false:
        return False, "env"
    raise ConfigError(f"{name}={raw!r}: expected one of {', '.join(TRUE + FALSE + tuple(extra_false))}")


def env_number(name, default, kind=float, minimum=None):
    """A numeric switch; refuses what does not parse. Returns (value, source)."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default, "default"
    try:
        value = kind(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r}: expected a {kind.__name__}") from None
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name}={raw!r}: must be at least {minimum}")
    return value, "env"


def env_text(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default, "default"
    return raw, "env"


# ---------------------------------------------------------------------------
# hardware

def probe_hardware(device):
    """What the engine runs on, without building or loading anything."""
    hw = {"device": str(device), "cpu": platform.processor() or platform.machine(),
          "cpu_threads": os.cpu_count(), "simd": _simd()}
    try:
        hw["ram_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    hw["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        if platform.system() == "Darwin":
            import subprocess
            try:
                hw["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                           text=True, timeout=2).stdout.strip() or hw["cpu"]
            except (OSError, subprocess.SubprocessError):
                pass
    if str(device).startswith("cuda"):
        import torch
        if torch.cuda.is_available():
            idx = torch.device(device).index or 0
            props = torch.cuda.get_device_properties(idx)
            hw.update(gpu=props.name, gpu_memory_bytes=props.total_memory,
                      gpu_capability=f"{props.major}.{props.minor}", gpu_count=torch.cuda.device_count())
    return hw


def _simd():
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "neon"
    try:
        with open("/proc/cpuinfo") as f:
            flags = next((line for line in f if line.startswith("flags")), "")
        return "avx512" if " avx512f" in flags else ("avx2" if " avx2" in flags else "sse")
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# implementations

@dataclasses.dataclass(frozen=True)
class Impl:
    """One physical implementation of a logical op."""
    op: str
    name: str
    fn: object
    devices: tuple            # device types it runs on ("cpu", "cuda")
    numerics: str             # "bit-identical" or "rounding", against the reference computation
    when: str                 # human description of when it applies
    applies: object = None    # applies(plan, **ctx) -> bool, beyond the device check
    reference: bool = False   # the reference computation: always applicable, on any device


REGISTRY = collections.defaultdict(list)  # op -> [Impl], in preference order


def implementation(op, name, devices, numerics, when, applies=None, reference=False):
    """Register fn as an implementation of op. Registration order is the
    preference order: the first one that applies is selected."""
    def register(fn):
        REGISTRY[op].append(Impl(op, name, fn, tuple(devices), numerics, when, applies, reference))
        return fn
    return register


@dataclasses.dataclass
class Choice:
    value: object
    source: str
    alternatives: tuple = ()
    doc: str = ""


# ---------------------------------------------------------------------------
# the plan

class Plan:
    """The physical plan of one engine. See the module docstring."""

    def __init__(self, device, compile_decode=False, external_dir=None, triton_available=None):
        self.device = str(device)
        self.device_type = self.device.split(":")[0]
        self.hardware = probe_hardware(device)
        self.choices = {}
        self.notes = []
        self._selections = collections.Counter()
        self._lock = threading.Lock()
        cuda = self.device_type == "cuda"
        if triton_available is None:
            from . import triton_ops
            triton_available = triton_ops.available()
        self._set("triton", triton_available, "probe", (True, False),
                  "Triton importable with a GPU: the GPU kernels (activation quantization, fp8 GEMV, Sinkhorn, head)")

        # the CPU kernels (csrc/): built on first use; off means the aten paths
        on, src = env_flag("VXPU_KERNELS", True, extra_false=("aten",))
        self._set("kernels", on, src, (True, False), "the fused C++ CPU kernels (fp4 experts, fp8 linears)")
        on, src = env_flag("VXPU_FUSED_EXPERTS", True)
        self._set("fused_experts", on, src, (True, False),
                  "one C++ call per decode layer's experts, instead of three kernel calls with the module's SwiGLU between")
        threads, src = env_number("VXPU_KERNEL_THREADS", 0, int, minimum=0)
        self._set("kernel_threads", threads or "torch's thread count", src, (),
                  "threads for the CPU kernels (read by the extension; 0 means torch's count)")

        # decode compilation: the split at host ops and the compile mode
        split, src = env_flag("VXPU_SPLIT_DECODE", True)
        self._set("compile_decode", bool(compile_decode), "argument", (True, False), "torch.compile the decode graph")
        self._set("split_decode", split and bool(compile_decode), src if compile_decode else "requires compile_decode",
                  (True, False), "cut the decode graph at its host ops; each dense segment compiled on its own")
        mode_default = ("reduce-overhead" if split else "max-autotune-no-cudagraphs") if cuda else "default"
        mode, src = env_text("VXPU_COMPILE_MODE", mode_default)
        self._set("compile_mode", mode, src, ("reduce-overhead", "max-autotune-no-cudagraphs", "default"),
                  "torch.compile mode (reduce-overhead gives each split segment a CUDA graph)")
        on, src = env_flag("VXPU_SPLIT_COMPILE", True)
        self._set("split_compile", on, src, (True, False),
                  "compile the dense segments (off: the split alone, segments eager, a check of the cut)")
        on, src = env_flag("VXPU_DIRECT_HOST_OPS", True)
        self._set("direct_host_ops", on, src, (True, False),
                  "eager host-op segments call the op's implementation directly, not through torch's dispatcher")

        # where the external tensors live
        if external_dir:  # an explicit argument wins over the environment, as it always has
            ext_dir, src = external_dir, "argument"
        else:
            ext_dir, src = env_text("VXPU_EXTERNAL_DIR", None)
        self._set("external_dir", ext_dir, src, (), "directory for memory-mapped external tensors")
        in_ram, src = env_flag("VXPU_EXPERTS_IN_RAM", False)
        placement = "ram" if (in_ram or not ext_dir) else "mmap"
        self._set("experts_placement", placement, src if in_ram else ("default" if not ext_dir else "external_dir"),
                  ("ram", "mmap"), "expert banks held in RAM, or memory-mapped from external_dir (the page cache as the RAM tier)")

        # the GPU's bf16 copies of fp8 weights (prefill, and decode without Triton)
        on, src = env_flag("VXPU_BF16_WEIGHT_CACHE", True)
        self._set("bf16_weight_cache", on, src, (True, False),
                  "keep a bf16 copy of each dense fp8 weight on the GPU (faster prefill, twice the dense weights' memory)")
        gb, src = env_number("VXPU_BF16_CACHE_HEADROOM_GB", 4.0, float, minimum=0)
        self._set("bf16_cache_headroom_gb", gb, src, (), "device memory the bf16 copies must leave free")

    def _set(self, name, value, source, alternatives=(), doc=""):
        self.choices[name] = Choice(value, source, tuple(alternatives), doc)

    def __getitem__(self, name):
        return self.choices[name].value

    def note(self, text):
        """A runtime event worth reporting with the plan (once each)."""
        with self._lock:
            if text not in self.notes:
                self.notes.append(text)

    # ---- dispatch

    def candidates(self, op, device_type):
        return [impl for impl in REGISTRY[op] if impl.reference or device_type in impl.devices]

    def select(self, op, device_type, key=None, **ctx):
        """The implementation of op to run for this call: the first
        registered one that runs on device_type and applies under this plan
        and ctx. key (input shapes, say) is recorded with the selection."""
        for impl in REGISTRY[op]:
            # the reference computation runs anywhere torch does
            if impl.reference or (device_type in impl.devices and (impl.applies is None or impl.applies(self, **ctx))):
                with self._lock:
                    self._selections[(op, impl.name, key)] += 1
                return impl
        raise RuntimeError(f"no implementation of {op} for {device_type} ({ctx})")

    # ---- explain

    def explain(self, ops=None):
        """The plan as JSON-able data. ops limits the op table to the ops an
        artifact calls (default: every registered op)."""
        names = sorted(ops) if ops is not None else sorted(REGISTRY)
        selections = collections.defaultdict(list)
        with self._lock:
            for (op, impl, key), n in sorted(self._selections.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
                selections[op].append({"impl": impl, "inputs": key, "dispatches": n})
        table = {}
        for op in names:
            table[op] = {
                "implementations": [
                    {"name": i.name, "devices": list(i.devices), "numerics": i.numerics, "when": i.when,
                     "on_this_device": i.reference or self.device_type in i.devices}
                    for i in REGISTRY[op]],
                "selected": selections.get(op, []),
            }
        return {
            "format": "vxpu-plan/1",
            "hardware": self.hardware,
            "choices": {k: {"value": c.value, "source": c.source, "alternatives": list(c.alternatives), "doc": c.doc}
                        for k, c in self.choices.items()},
            "ops": table,
            "notes": list(self.notes),
        }

    def summary(self, ops=None):
        """A compact, human-readable form of explain() for logs."""
        e = self.explain(ops)
        hw = e["hardware"]
        lines = [f"plan on {hw['device']}: {hw.get('gpu', hw['cpu'])}"
                 + (f", {hw['gpu_memory_bytes'] / 1e9:.0f} GB" if "gpu_memory_bytes" in hw else "")
                 + f", {hw.get('ram_bytes', 0) / 1e9:.0f} GB RAM, {hw['cpu_threads']} CPU threads, {hw['simd']}"]
        lines.append("  " + ", ".join(f"{k}={c['value']}" + ("" if c["source"] in ("default", "probe") else f" ({c['source']})")
                                      for k, c in e["choices"].items()))
        for op, row in e["ops"].items():
            here = [i["name"] for i in row["implementations"] if i["on_this_device"]]
            got = sorted({s["impl"] for s in row["selected"]})
            lines.append(f"  {op}: {' > '.join(here) or '-'}" + (f"   selected: {', '.join(got)}" if got else ""))
        lines += [f"  note: {n}" for n in e["notes"]]
        return "\n".join(lines)


_default = {}


def default_plan(device_type):
    """The plan for op calls made outside any engine (direct calls in tests
    and tools): built from the environment once per device type."""
    plan = _default.get(device_type)
    if plan is None:
        plan = _default[device_type] = Plan(device_type)
    return plan


# ---------------------------------------------------------------------------
# EXPLAIN without loading weights

def artifact_ops(artifact_dir):
    """The vxpu ops an artifact's graphs call."""
    import torch
    from . import ops as _ops  # noqa: F401  (registers the ops the graphs reference)
    found = set()
    for name in ("prefill.pt2", "decode.pt2"):
        path = os.path.join(artifact_dir, name)
        if not os.path.exists(path):
            continue
        ep = torch.export.load(path)
        for node in ep.graph.nodes:
            target = str(node.target)
            if node.op == "call_function" and target.startswith("vxpu."):
                found.add("vxpu::" + target.split(".")[1])
    return found


def main(argv=None):
    p = argparse.ArgumentParser(description="Explain the physical plan for an artifact, without loading weights.")
    p.add_argument("--artifact", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--compile", action="store_true", help="the plan for a compiled decode (the executor's GPU default)")
    p.add_argument("--json", action="store_true", help="print the full explain() JSON")
    args = p.parse_args(argv)
    # run as `python -m vxpu.plan` this file is __main__, a second copy of
    # the module: the ops register into vxpu.plan's registry, so use that one
    from . import ops as _ops  # noqa: F401  (registers the implementations)
    from . import plan as registered
    plan = registered.Plan(args.device, compile_decode=args.compile)
    used = registered.artifact_ops(args.artifact)
    print(json.dumps(plan.explain(used), indent=1) if args.json else plan.summary(used))


if __name__ == "__main__":
    main()
