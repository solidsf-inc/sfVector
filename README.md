# sfVector

Fresh MIT CUDA C++17 batched FP32 dot products and cosine similarity. Include
`src/kernel.cuh` and call `sfVector::dot(a, b, output, rows, columns, stream)`
or `sfVector::cosine(a, b, output, rows, columns, stream)`. Inputs contain
contiguous `[rows,columns]` device floats and output contains `[rows]`.
One 256-thread block reduces each row, and a bounded grid covers larger
batches. Operations enqueue on the supplied stream and return argument/launch
status; callers synchronize that stream to observe completion errors.

Dot products use FP32 FMA accumulation and a parallel FP32 reduction, so their
rounding differs from a serial sum. Cosine first normalizes each vector by
its maximum magnitude, then reduces dot and squared norms. This avoids
overflow or underflow caused only by very large or small finite input scales.
Either zero vector, including two zero vectors or empty vectors, has cosine
zero. Finite cosine is clamped to `[-1,1]`. The documented contract is finite
input; nonfinite input is not a valid similarity measurement. Dot results may
overflow FP32 when their mathematical magnitude is too large.

Negative dimensions, missing required pointers, unaddressable extents, and
exact output/input aliases are rejected. Partial overlapping ranges are
unsupported. Zero rows is a no-op. Positive rows with zero columns write zero
outputs and accept null inputs.

Portable source targets Hopper and Blackwell (`sm_90`, `sm_100`, `sm_103`,
`sm_120`, `sm_121`, including GB10), using ordinary CUDA cores. Nine numerical
and API-domain checks passed on GB10 with CUDA 13.0. For four rows of 1024
elements, dot took 2.768 us versus 27.058 us for four cuBLAS SDOT calls
(9.775x), and cosine took 4.686 us versus a parallel CUDA reference at
5.196 us (1.109x). These seven-sample CUDA-event medians exclude setup and
transfers; they qualify only these workloads on GB10.

`tests/check.cu` compares against scalar long-double dot/norm references on a
nondefault stream. It covers scalar rows, widths with tails, zero/parallel/
antiparallel vectors, opposing `1e30` and `1e-30` scales, empty rows/batches,
invalid arguments, aliases, and output sentinels. It emits one JSON verdict
with per-case results and fails on any failed case. A CUDA toolkit and GPU
are required.

Prior-source inventory: solidSF contains mixed-format warp dot/GEMV kernels,
solver dot reductions, and CPU cosine references with differing zero-vector
semantics. Exact locations are recorded in the private suite inventory. The
standalone operators here are newly written, with the zero-vector convention
stated above; no private implementation is copied. No vector-search index,
approximate nearest-neighbor algorithm, or model behavior is included.

## Standalone build

CUDA 13.0 or newer is required for the full target matrix.

```sh
make -j1
make test
# A single device family can be selected explicitly:
make ARCHS=90 BUILD_DIR=build-hopper
```

Default targets: H100/H200 (`sm_90`), B200 (`sm_100`), B300/GB300 (`sm_103`),
GB10 (`sm_121`), plus portable Hopper PTX. Do not add `--use_fast_math`;
that changes the numerical contract. Compile success and numerical
qualification are reported separately from performance benchmarking.

## Published source and GB10 measurements

The standalone source repository is [solidsf-inc/sfVector](https://github.com/solidsf-inc/sfVector). The [solidSF CUDA kernel catalog](https://solidsf.com/kernels) contains the formal report, all measured workloads and raw timing samples.

| Workload | SF median us | Reference median us | Reference / SF |
| --- | ---: | ---: | ---: |
| batched dot fp32; 4 x 1024 | 2.460 | 17.188 | 6.987x |
| cosine fp32; 4 x 1024 | 4.898 | 5.282 | 1.078x |

References: cuBLAS cublasSdot per row device result FP32; conventional CUDA CTA per row parallel scaled cosine.

These are seven-sample medians of alternating CUDA-event measurements on one GB10 at stock clocks. Setup, allocations, transfers, plan creation and destructive-input resets are excluded; stream timing can include API enqueue gaps. Ratios above one mean SF is faster; near-unity medians do not establish a meaningful advantage. No aggregate or universal speedup is claimed.

All 9 package numerical/API cases passed on GB10. All four cubin targets and compute_90 PTX were verified. H100, H200, B200, B300 and GB300 hardware runtime and performance remain unmeasured.
The batched-dot reference uses four cuBLAS SDOT calls for four rows.
