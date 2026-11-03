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

"""Check pod environment, caches, scratch, and page cache before a load (#511)."""

import argparse, ctypes, glob, hashlib, json, mmap, os, shutil, sys, tempfile
from .localblobs import _complete

TOKEN_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json", "spiece.model")


def check_blob_dir(manifest, env=os.environ):
    blob_dir, files = env.get("VXPU_BLOB_DIR"), manifest.get("files", {})
    names = [e.get("name", sha) for sha, e in files.items()]
    if not blob_dir: return "FAIL", f"VXPU_BLOB_DIR unset (missing {len(names)} files: {', '.join(names)})"
    missing = [e.get("name", sha) for sha, e in files.items() if not _complete(os.path.join(blob_dir, sha), e["size"])]
    return ("FAIL", f"missing {len(missing)} of {len(files)} files: {', '.join(missing)}") if missing else ("PASS", f"{len(files)} files present in {blob_dir}")


def _external_paths(manifest, binding, ext_dir):
    if not ext_dir: return []
    mm = os.path.join(ext_dir, hashlib.sha256(json.dumps(manifest.get("source", {}), sort_keys=True).encode()).hexdigest()[:16])
    ext, tensors = (binding or {}).get("external") or manifest.get("external") or {}, manifest.get("tensors", {})
    paths = [p if os.path.exists(p) else os.path.join(ext_dir, f"experts.{l}.{n}")
             for l, sps in ext.get("experts", {}).items() for n in sps for p in [os.path.join(mm, f"experts.{l}.{n}")]]
    for sp in ext.get("engram", {}).values():
        for k in ("weight", "scale"):
            if k in sp and sp[k] in tensors:
                r = tensors[sp[k]]
                p = os.path.join(mm, hashlib.sha256(f"{r['file_sha256']}:{r['offset']}:{r['length']}".encode()).hexdigest())
                paths.append(p if os.path.exists(p) else os.path.join(ext_dir, os.path.basename(p)))
    return paths


def check_external_dir(manifest, binding=None, env=os.environ):
    ext_dir = env.get("VXPU_EXTERNAL_DIR")
    paths = _external_paths(manifest, binding, ext_dir or "/dev/null")
    if not paths: return "PASS", "no external tensors in artifact"
    if not ext_dir: return "FAIL", f"VXPU_EXTERNAL_DIR unset (missing {len(paths)} files)"
    missing = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    return ("FAIL", f"missing {len(missing)} files: {', '.join(missing)}") if missing else ("PASS", f"{len(paths)} files present in {ext_dir}")


def check_tokenizer(manifest, env=os.environ):
    hf_home, repo = env.get("HF_HOME"), manifest.get("source", {}).get("repo", "")
    if not hf_home: return "FAIL", "HF_HOME unset"
    if os.path.isdir(repo): return ("PASS" if any(os.path.exists(os.path.join(repo, f)) for f in TOKEN_FILES) else "FAIL"), f"repo {repo}"
    norm = repo.replace("/", "--")
    found = any(any(f in TOKEN_FILES for f in fs) for r, _, fs in os.walk(hf_home) if not repo or norm in r)
    return ("PASS", f"tokenizer present in {hf_home} for {repo}") if found else ("FAIL", f"tokenizer files for {repo} missing from {hf_home}")


def check_environment(env=os.environ):
    if env.get("HF_HUB_OFFLINE") not in ("1", "true", "True"): return "FAIL", "HF_HUB_OFFLINE is not 1"
    scratch = os.path.realpath(env.get("VXPU_SCRATCH_DIR") or ("/mnt/ssd" if os.path.isdir("/mnt/ssd") else "/tmp"))
    bad = [f"{k}={env.get(k)}" for k in ("TMPDIR", "VXPU_AOTI_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
           if not env.get(k) or not os.path.realpath(env.get(k)).startswith(scratch)]
    return ("FAIL", f"not under scratch {scratch}: {', '.join(bad)}") if bad else ("PASS", f"offline and paths under {scratch}")


def check_scratch(env=os.environ):
    scratch = env.get("VXPU_SCRATCH_DIR") or ("/mnt/ssd" if os.path.isdir("/mnt/ssd") else tempfile.gettempdir())
    if not os.path.exists(scratch): return "FAIL", f"scratch directory {scratch} missing"
    u, tmpdir, strays = shutil.disk_usage(scratch), env.get("TMPDIR", scratch), 0
    free_gb, used_pct = u.free / 1e9, (u.used / u.total) * 100
    if os.path.isdir(tmpdir):
        for e in os.scandir(tmpdir):
            if e.name.startswith(("tmp", "torch_", "aoti_")) or ".partial." in e.name:
                try: strays += sum(f.stat().st_size for f in os.scandir(e.path) if f.is_file()) if e.is_dir() else e.stat().st_size
                except OSError: pass
    msg = f"{free_gb:.1f} GB free ({used_pct:.1f}% used), {strays / 1e9:.1f} GB temporaries"
    return ("FAIL" if (used_pct >= 95.0 or free_gb < 10) else "WARN" if (strays > 5e9 or used_pct >= 90.0) else "PASS"), msg


def check_caches(env=os.environ):
    aoti = env.get("VXPU_AOTI_CACHE_DIR") or os.path.expanduser("~/.cache/vxpu/aoti")
    if not os.path.isdir(aoti): return "FAIL", f"AOTI cache {aoti} missing"
    try:
        probe = os.path.join(aoti, f".probe.{os.getpid()}")
        with open(probe, "w"): pass
        os.remove(probe)
    except OSError as e: return "FAIL", f"AOTI probe not writable: {e}"
    for name, var, dflt in (("inductor", "TORCHINDUCTOR_CACHE_DIR", "~/.cache/torch/inductor"), ("triton", "TRITON_CACHE_DIR", "~/.triton/cache")):
        if not os.path.isdir(env.get(var) or os.path.expanduser(dflt)): return "FAIL", f"{name} cache missing"
    return "PASS", "AOTI probe writable, inductor and Triton caches present"


def check_page_cache(manifest, binding=None, env=os.environ, samples=1000):
    paths = [p for p in _external_paths(manifest, binding, env.get("VXPU_EXTERNAL_DIR")) if os.path.exists(p)]
    if not paths: return "PASS", "no bank table files to check"
    try:
        libc = ctypes.CDLL(None)
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    except Exception: return "WARN", "mincore unavailable"
    resident, total, vec = 0, 0, (ctypes.c_char * 1)()
    for p in paths:
        sz = os.path.getsize(p)
        if not sz: continue
        pages, n = (sz + 4095) // 4096, min(samples, (sz + 4095) // 4096)
        with open(p, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            resident += sum(ord(vec[0]) & 1 for i in range(n) if libc.mincore(addr + i * (pages // n) * 4096, 4096, vec) == 0)
            total += n
    if not total: return "PASS", "no pages sampled"
    pct = (resident / total) * 100
    msg = f"{pct:.1f}% resident across {len(paths)} bank files"
    return ("WARN", f"cold node, only {msg} (#504)") if pct < 90.0 else ("PASS", msg)


def check_gpus(env=os.environ):
    locks = glob.glob(env.get("VXPU_LOCK_PATTERN", "/root/hero/h15/LOCK-*"))
    lock_msg = f"; held by {locks[0]}" if locks else ""
    try:
        import torch
        if not torch.cuda.is_available(): return "WARN", f"CUDA unavailable{lock_msg}"
        cnt = torch.cuda.device_count()
        gbs = [f"{torch.cuda.mem_get_info(i)[0] / 1e9:.1f} GB" for i in range(cnt)]
        return ("WARN" if locks else "PASS"), f"{cnt} GPUs ({', '.join(gbs)} free){lock_msg}"
    except Exception as e: return ("WARN" if locks else "PASS"), f"GPU check skipped ({e}){lock_msg}"


def run_doctor(artifact_dir, env=os.environ, log=print):
    art = artifact_dir or env.get("VXPU_ARTIFACT_DIR", ".")
    mpath = os.path.join(art, "manifest.json") if os.path.isdir(art) else art
    with open(mpath) as f: manifest = json.load(f)
    bpath = os.path.join(os.path.dirname(mpath), "binding.json")
    binding = json.load(open(bpath)) if os.path.exists(bpath) else None
    failed = False
    for name, fn in [("blob_dir", lambda: check_blob_dir(manifest, env)),
                     ("external_dir", lambda: check_external_dir(manifest, binding, env)),
                     ("tokenizer", lambda: check_tokenizer(manifest, env)),
                     ("environment", lambda: check_environment(env)),
                     ("scratch", lambda: check_scratch(env)),
                     ("caches", lambda: check_caches(env)),
                     ("page_cache", lambda: check_page_cache(manifest, binding, env)),
                     ("gpus", lambda: check_gpus(env))]:
        status, msg = fn()
        log(f"{status:<4}  {name}: {msg}")
        if status == "FAIL": failed = True
    return 1 if failed else 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Check pod environment and caches before a load")
    p.add_argument("artifact", nargs="?", default=".", help="artifact directory")
    sys.exit(run_doctor(p.parse_args(argv).artifact))


if __name__ == "__main__": main()
