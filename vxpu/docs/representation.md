# The graph representation: what it must do, and why it is pt2

vXPU builds a model's graph on one machine (a laptop, CI: no GPU, no
weights) and executes it on another (a GPU or CPU node in a cluster).
The thing that travels between them is the subject of this note.

## The requirement

Loading a pretrained model in PyTorch builds a module tree; the graph
exists only while `forward` runs, created by Python on each call. Our
representation must capture that graph such that executing it on the
executor is **behaviourally equivalent to rebuilding the module tree
there and running it**, with three clarifications:

1. **Same computation, not merely same answer.** A reformulation that
   produces the same function at a different cost (every expert on every
   token, masked by zero weights) is not equivalent. Routing must stay
   sparse, shapes that are data-dependent must be representable, and
   prompt length must be a dynamic dimension.
2. **No miracles.** If the graph needs a CUDA kernel it should fail on a
   CPU executor, and if it needs more memory than an L4 has it should
   fail on an L4, exactly as the local run would. Device-compatibility
   rewrites in the executor are a convenience, not part of the contract.
3. **Weightless.** The graph is captured without weights; the artifact
   binds every tensor to a content-addressed range of the published
   checkpoint, bit for bit, including a model maker's quantized tensors.

Things we also want from it, in order: the executor must not need the
model's Python code; module boundaries should survive so an engine can
recognize a known layer (a Qwen3 MoE block, a Gemma attention block) and
substitute its own kernel while running unknown modules as-is; and the
same representation should serve optimization (fusion, compilation) and,
eventually, training.

## Why not a module-tree format of our own

A module tree alone is nodes without edges: the wiring lives in the
`forward` code of composite modules, not in the leaves. So a shipped
module tree either requires the same libraries on the executor (code
shipping, version-pinned, exact by construction) or must be lowered to a
graph by tracing. The first is a legitimate *executor method* (and the
way a vLLM or llama.cpp backend will work), but it is not a
representation of the graph. The second is what `torch.export` already
does, and it keeps the tree.

## What pt2 gives us (checked against torch 2.13 and 2.14)

- **Aten-level graph, no Python.** The executor runs ops, not model
  code. The .pt2 archive is a declarative graph plus tensor metadata
  (the only pickle in it is the sample inputs, which we can drop).
- **Dynamic dimensions.** Prompt length is a symbolic dimension of the
  prefill graph.
- **Data-dependent shapes.** `torch.nonzero`-style routing exports with
  unbacked sizes, serializes, reloads and runs. We verified this with an
  eager-style per-expert routing loop over an fp8 expert bank; it is not
  what we ship (see below), but the format can carry it.
- **Variable-size groups inside an op.** `grouped_mm` takes static-shape
  inputs and per-group offsets, so MoE prefill is sparse with static
  shapes around it. It has a CPU kernel in this torch.
- **Module boundaries survive.** Every aten node carries
  `nn_module_stack` (the module path and class that produced it), and
  `torch.export.unflatten` rebuilds a module hierarchy from the graph.
  This is the "recognize known layers, run unknown ones as a graph"
  hybrid, already present in the format.
- **A compiler input.** The exported graph is what inductor consumes,
  so engine-side fusion is `torch.compile` of the loaded program. This
  is what makes reference semantics affordable: the fp8 linear in the
  graph is cast, scale, matmul, which read alone is 2x slower than bf16
  on an L4; with inductor's prologue fusion the executor folds the
  dequantization into the GEMM's operand load and the same graph runs
  at the fp8 bandwidth floor, faster than bf16 and 3x faster than
  transformers' own fp8 kernel path on that GPU. The artifact never
  changed; the executor's compiler settings did.
- **Training.** The unflattened module is an ordinary module of aten ops
  and runs under autograd; export also has a training IR. Plausible,
  not yet exercised here.

## What pt2 cannot carry, and what we do about it

- **Tensor subclasses with Python state** (bitsandbytes' `Params4bit`).
  Not serializable. The fix belongs in the library (plain tensors plus
  a custom op), not in the format.
- **Python-side state objects** (transformers' growing `DynamicCache`).
  We represent the cache as tensors instead: a static cache today, a
  paged one planned. Each graph mutates its state buffers in place.
- **The generation loop.** A graph is one `forward`; the loop and the
  sampling rules are a program. The executor runs transformers'
  loop semantics over the two graphs (prefill, decode).
- **Version coupling.** A .pt2 pins the exporting torch version (the
  format promises newer-loads-older; same-version is what we test).

## Fixing libraries instead of working around them

Whether a blocker is "a pt2 limitation" or "a transformers limitation"
is academic except for one thing: where the fix goes. The format is hard
to change; a library is not. So when a library stands between a model
and a faithful export, we carry a small, upstreamable patch in a fork
under `third_party/` (a git submodule branched from the pinned release)
rather than reimplement the module in vXPU.

The first case: transformers' fp8 modules (`FP8Linear`, `FP8Experts`)
had no computation at all off the GPU, only dispatch to CUDA kernels, so
on a GPU-less machine there was no fp8 graph to capture and the first
vXPU artifact used fp8 modules of our own. The fork adds a plain-aten
reference path (dequantize the block, matmul; grouped and batched
variants for the experts), after which transformers' own modules export
weightlessly, match the dequantized reference exactly, and run an fp8
checkpoint on a CPU without `dequantize=True`. vXPU's fp8 modules were
deleted. The exported graph is now transformers' computation, with its
own routing (`grouped_mm` for prefill, `batched_mm` for decode).

## Where we deviated, and the correction

The first MoE fp8 artifact ran prefill *densely* (every expert on every
token, masked). That met requirement 1 in answer but not in computation,
and was a shortcut, not a limit of the format: transformers' own eager
experts use data-dependent `nonzero` routing, which transformers marks
as non-exportable, so the fp8 expert bank was written fresh and the
simplest static formulation was chosen without checking whether
`grouped_mm` existed on the CPU. It does. Prefill now uses the same
grouped-matmul formulation transformers uses for its exportable experts
implementation: pairs sorted by expert, cumulative counts as offsets,
one grouped matmul per projection. Compute is proportional to tokens
times top-k, and the result matches transformers' eager MoE exactly on
a small model.

The executor-side rewrite of integer `histc` to `bincount` on the CPU
is the kind of convenience requirement 2 says we should not rely on. The
fork fixes its cause (the grouped experts now count in float off the
GPU, meta included), so new artifacts do not need it and the rewrite
no longer touches them; it stays for artifacts exported before that.

## What Qwen3.8-Flash-Next taught us

The hybrid model was the test of the format, and pt2 passed it, with
every fix landing in the library rather than the artifact:

- **Data-dependent loops.** The chunked delta rule runs a recurrence
  over a number of chunks that depends on the prompt length. A Python
  loop specializes the length; the `scan` and `while_loop` higher-order
  ops export, serialize and reload with a symbolic trip count. The
  fork uses `while_loop` with per-chunk `index_select`, because strided
  chunk views of a symbolic-length tensor make the shape reasoner
  case-split on "could this be one chunk?".
- **Python control flow on data.** The indexer picked tokens with a
  `nonzero` loop per query. With a static cache the visible set is a
  known function of the position, so the same selection is a masked
  top-k over all blocks; equal to the loop to float32 rounding.
- **Growing Python state.** Full position tables lived on the cache
  object and grew by concatenation; with a static cache they are the
  table for every slot, indexed by `cache_position`.
- **0/1 specialization.** Branches on the length (`if seq < kernel`),
  `-1` in reshapes next to the length, and contiguity requests on
  transposed activations all pin the length to the example's range.
  Non-strict export plus `backed_size_oblivious` reasoning, and
  removing those patterns in the fork, leave one guard we accept: the
  prefill graph starts at two tokens, and a one-token prefill is the
  decode graph.
- **Two names for one tensor.** Non-strict export lifts a cache tensor
  reached through the cache object as a constant even when it is also
  a wrapper buffer. The binding records the alias; the executor binds
  both names to one tensor. Separate tensors would not be a cache.
- **Checkpoint layout is the loader's business.** Key renaming and the
  recipes that merge per-expert projections into a bank or 128 shards
  into a table come from transformers' conversion mapping, replayed on
  the manifest; nothing in vXPU knows a model's naming.

## Decision

pt2 is the representation. Our own module-tree format would reinvent
the parts of pt2 that already exist (module stacks, unflattening) and
inherit every tracing problem unchanged. Effort goes into (a) exporting
the model maker's computation faithfully, including quantized tensors
and sparse routing, (b) fixing libraries whose tensors cannot be
represented, and (c) executor methods, where "rebuild the module tree
from the same libraries" and "hand the recognized model to vLLM" are
peers of "run the graph".
