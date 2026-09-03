# vXPU: run PyTorch models on remote accelerators

vXPU lets you take a PyTorch model, export it on a machine with **no
GPU, no CUDA, and no downloaded weights**, and run it on a Kubernetes
cluster that has the accelerators — shipping only a small, portable,
content-addressed artifact.

```sh
# 1. Export (thin client: meta device, no weights ever downloaded)
python -m vxpu.export google/gemma-4-E4B-it -o gemma-e4b/

# 2. Ask (creates the executor pod on demand, ships the artifact,
#    weights rehydrate on the executor from content-addressed refs)
vxpu ask --artifact gemma-e4b/ "Is the sky blue?"
```

The exported artifact for a 16 GB model is ~60 MB. For a 689 GB model
it is ~25 MB of manifest — the artifact size scales with the
*architecture*, not the weights.

## How it works

```
 thin client (laptop, CI)                 executor pod (GPU/CPU node)
┌──────────────────────────────┐  gRPC   ┌──────────────────────────────┐
│ meta-device instantiation    │ ──────▶ │ verify + cache weights by    │
│ torch.export (prefill+decode)│ artifact│   content hash (range reads) │
│ manifest: tensor →           │  ~60MB  │ recompute derived tensors    │
│   (sha256, offset, len,      │         │ zero-init session state      │
│    dtype, shape)             │ ◀────── │ torch.compile decode once    │
│ chat over sessions           │  tokens │ run the token loop           │
└──────────────────────────────┘         └──────────────────────────────┘
```

Three ideas carry the design:

1. **Weightless export.** torch's meta device instantiates a model's
   shapes without storage, and `torch.export` captures executable
   graphs from it. A laptop can export a model of any size. Two graphs
   are captured: a dynamic-length `prefill` and a constant-shape
   `decode` — constant shapes mean the executor compiles it exactly
   once and reuses it for every token.

2. **Content-addressed weights.** The manifest binds every tensor to
   `(file_sha256, offset, length, dtype, shape)`, built entirely from
   repository metadata (the Hub serves per-file sha256; safetensors
   headers are range-fetched). Executors pull each tensor with one
   HTTP range request into a local cache: cold loads stream, warm
   loads download nothing, and the sha256 of the manifest is the
   model's identity — `LoadModel` with a matching digest is a no-op.

3. **A three-way tensor classification.** Every tensor the graphs
   reference is `bound` (a weight reference), `derived` (computed from
   config at load time: RoPE tables via transformers' own formula
   registry, embedding scales — never stored in checkpoints), or
   `state` (per-session KV cache, zero-initialized, mutated in place
   by the graphs). The classification is also the serving
   architecture: bound/derived are shared read-only across sessions;
   a session *is* its state tensors. Multi-turn chat prefills only the
   new suffix tokens against the session's existing cache.

The executor applies device-specific compatibility passes to the
shipped graph at load time (e.g. rewriting `histc` — no CPU kernel for
integer inputs — to its exact `bincount` equivalent), so one artifact
serves heterogeneous executors: the graph carries reference semantics;
each engine adapts it to its hardware.

## Layout

```
proto/        Executor gRPC API (LoadModel / NewSession / Chat)
python/vxpu/  export (thin client) + server (executor): manifest,
              export, rehydrate, engine, server
cmd/vxpu/     Go CLI: no Python/torch — ships artifacts, creates the
              executor pod on demand, port-forwards, chats
images/       executor container image
```

## Building

```sh
# CLI
go build ./cmd/vxpu/

# Executor image (from vxpu/):
gcloud builds submit --config cloudbuild.yaml \
    --substitutions _IMAGE=gcr.io/$PROJECT/vxpu-executor:v1 .
export VXPU_EXECUTOR_IMAGE=gcr.io/$PROJECT/vxpu-executor:v1
```

## Verified behavior

- Executor logits are bitwise-identical to `from_pretrained` for the
  same device (verified cross-OS and cross-architecture), and cached
  generation is token-for-token identical to Hugging Face `generate`
  with a static cache.
- Gemma-4-E4B (hybrid local/global attention, p-RoPE, 16 GB bf16)
  exports with every tensor classified and generates coherently
  through the executor on an L4: ~22 tok/s steady-state, plus a ~33 s
  one-time `torch.compile` on the first decode step. That ~22 tok/s is
  close to the L4's bf16 memory-bandwidth roofline (~300 GB/s ÷ ~16 GB
  read per token ≈ 19 tok/s) — LLM decode is bandwidth-bound, so the
  lever for more speed is quantization or a higher-bandwidth card, not
  the pipeline. (The same L4 served a 4-bit 26B-A4B at ~60 tok/s in the
  experiment, reading ~4x fewer bytes per token.)
- A 26B MoE (53 GB) exports to a 20 MB graph and executes on a
   240 GB-RAM CPU node — the artifact is hardware-agnostic; placement
  is the executor's decision.

## Status and roadmap

vXPU is an early extraction of the `experiments/model-manifest`
work — the experiment retains the full research trail.

Known simplifications / tasks for the immediate roadmap are:
- Tokenizer/processor files ride alongside the manifest rather than
  in it; executors currently fetch them from the source repo.
- Artifacts travel inside the LoadModel RPC; OCI registries
  (per-tensor layers, signed manifests) are the intended distribution.
- One model per executor; engine-backed executors (vLLM behind the
  same proto) for architectures whose graphs are not natively
  runnable (quantized fused-MoE) are planned.
- `.pt2` graphs pin the exporting torch version (the serialization
  format promises newer-loads-older; same-version is what we test).
- A shared paged KV pool is designed (see the experiment's
  PLAN-paged-rewrite.md) to replace fixed-shape per-session static caches.
