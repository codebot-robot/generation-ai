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

"""The hand-written CPU kernels and native fetcher (csrc/), built on first use.

``load()`` compiles the CPU kernels extension with torch's JIT builder into
``TORCH_EXTENSIONS_DIR`` (the executor image warms that cache at build
time, so a pod never compiles) and caches the module. ``available()``
says whether a compiler and headers were there; the callers fall back to
the plain-aten path when they were not, so a machine without a
toolchain still runs, slowly.

``load_fetch()`` compiles the native libcurl fetcher extension into its
own module (``vxpu_fetch``) so that libcurl configuration or build issues
never affect the CPU kernels. ``has_fetch()`` reports whether the native
fetcher is available; callers fall back to Python requests otherwise.

``VXPU_KERNELS=off`` disables the CPU kernels (forcing the aten path).
``VXPU_FETCH=off`` disables the native libcurl fetcher (forcing requests).
"""

import os
import threading

_lock = threading.Lock()
_module = None
_error = None
_enabled = True

_fetch_lock = threading.Lock()
_fetch_module = None
_fetch_error = None

KERNEL_SOURCES = [
    os.path.join(os.path.dirname(__file__), "csrc", "fp4_kernels.cpp"),
]

FETCH_SOURCES = [
    os.path.join(os.path.dirname(__file__), "csrc", "fetch.cpp"),
]


def _curl_flags():
    """Detect libcurl and return (extra_cflags, extra_ldflags), or None if unavailable."""
    if os.environ.get("VXPU_FETCH", "").lower() in ("0", "off", "false", "requests", "fallback"):
        return None
    import shutil
    import subprocess

    cflags = ["-DHAVE_LIBCURL"]
    ldflags = []

    # 1. Try curl-config
    if shutil.which("curl-config") is not None:
        try:
            res_c = subprocess.run(["curl-config", "--cflags"], capture_output=True, text=True)
            res_l = subprocess.run(["curl-config", "--libs"], capture_output=True, text=True)
            if res_c.returncode == 0 and res_l.returncode == 0:
                cf = [f for f in res_c.stdout.strip().split() if f]
                lf = [f for f in res_l.stdout.strip().split() if f]
                return cflags + cf, ldflags + lf
        except Exception:
            pass

    # 2. Try pkg-config
    if shutil.which("pkg-config") is not None:
        try:
            res_c = subprocess.run(["pkg-config", "--cflags", "libcurl"], capture_output=True, text=True)
            res_l = subprocess.run(["pkg-config", "--libs", "libcurl"], capture_output=True, text=True)
            if res_c.returncode == 0 and res_l.returncode == 0:
                cf = [f for f in res_c.stdout.strip().split() if f]
                lf = [f for f in res_l.stdout.strip().split() if f]
                return cflags + cf, ldflags + lf
        except Exception:
            pass

    # 3. Fall back to standard include paths
    for inc in ["/usr/include", "/usr/local/include", "/opt/homebrew/include"]:
        if os.path.exists(os.path.join(inc, "curl", "curl.h")):
            return cflags + [f"-I{inc}"], ldflags + ["-lcurl"]

    return None


def set_enabled(on):
    """Switch the kernels off (and back on) at runtime: the ops then take
    the plain-aten paths. For A/B comparisons on one loaded engine."""
    global _enabled
    _enabled = bool(on)


def load():
    """The CPU kernels extension module, or None if it cannot be built here
    (or the kernels are switched off)."""
    global _module, _error
    if not _enabled:
        return None
    if _module is not None or _error is not None:
        return _module
    with _lock:
        if _module is not None or _error is not None:
            return _module
        from .plan import env_flag
        if not env_flag("VXPU_KERNELS", True, extra_false=("aten",))[0]:
            _error = "disabled by VXPU_KERNELS"
            return None
        try:
            from torch.utils.cpp_extension import load as jit_load
            import sys
            cflags, ldflags = ["-O3"], []
            if sys.platform != "darwin":  # gcc/clang with an OpenMP runtime; Apple clang has none
                cflags.append("-fopenmp")
                ldflags.append("-fopenmp")
            _module = jit_load(name="vxpu_fp4_kernels", sources=KERNEL_SOURCES,
                               extra_cflags=cflags, extra_ldflags=ldflags, verbose=False)
        except Exception as e:  # noqa: BLE001  (no compiler, no headers: the aten path remains)
            _error = f"{type(e).__name__}: {e}"
            print(f"[vxpu] CPU kernels unavailable, using the plain-aten path: {_error.splitlines()[0][:200]}", flush=True)
            return None
        print(f"[vxpu] CPU kernels loaded (avx512: {_module.has_avx512()}, neon: {_module.has_neon()})", flush=True)
        return _module


def available():
    return load() is not None


def why_not():
    """The build error, when available() is False."""
    load()
    return _error


def load_fetch():
    """The libcurl fetch extension module, or None if libcurl or a C++ toolchain
    is unavailable (or native fetch is switched off)."""
    global _fetch_module, _fetch_error
    if _fetch_module is not None or _fetch_error is not None:
        return _fetch_module
    with _fetch_lock:
        if _fetch_module is not None or _fetch_error is not None:
            return _fetch_module
        if os.environ.get("VXPU_FETCH", "").lower() in ("0", "off", "false", "requests", "fallback"):
            _fetch_error = "disabled by VXPU_FETCH"
            return None
        flags = _curl_flags()
        if flags is None:
            _fetch_error = "libcurl development headers or libraries not found"
            return None
        extra_cflags, extra_ldflags = flags
        try:
            from torch.utils.cpp_extension import load as jit_load
            extra_cflags = ["-O3"] + extra_cflags
            _fetch_module = jit_load(
                name="vxpu_fetch",
                sources=FETCH_SOURCES,
                extra_cflags=extra_cflags,
                extra_ldflags=extra_ldflags,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001
            _fetch_error = f"{type(e).__name__}: {e}"
            print(f"[vxpu] Native range fetcher unavailable: {_fetch_error.splitlines()[0][:200]}", flush=True)
            return None
        return _fetch_module


def has_fetch():
    """Whether the C++ libcurl range fetcher is available."""
    mod = load_fetch()
    return mod is not None and getattr(mod, "has_fetch", lambda: False)()


def why_not_fetch():
    """The build error for the fetch extension, when has_fetch() is False."""
    load_fetch()
    return _fetch_error


def fetch_ranges(ranges, **kwargs):
    """Fetch weight ranges using the native libcurl fetcher."""
    mod = load_fetch()
    if mod is None:
        raise RuntimeError(f"vxpu native fetcher unavailable: {_fetch_error}")
    return mod.fetch_ranges(ranges, **kwargs)
