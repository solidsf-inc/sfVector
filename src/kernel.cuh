// SPDX-License-Identifier: MIT
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace sfVector {
namespace detail {
template<bool Cosine>
static __global__ void reduce_kernel(const float* a, const float* b, float* output,
                                     int rows, int columns) {
    __shared__ float first[256];
    __shared__ float second[256];
    __shared__ float third[256];
    const int lane = threadIdx.x;
    for (uint64_t row = blockIdx.x; row < static_cast<uint64_t>(rows); row += gridDim.x) {
        const uint64_t base = row * columns;
        float scale_a = 1.0f, scale_b = 1.0f;
        if constexpr (Cosine) {
            float max_a = 0.0f, max_b = 0.0f;
            for (uint64_t i = lane; i < static_cast<uint64_t>(columns); i += 256) {
                max_a = fmaxf(max_a, fabsf(a[base + i]));
                max_b = fmaxf(max_b, fabsf(b[base + i]));
            }
            first[lane] = max_a;
            second[lane] = max_b;
            __syncthreads();
            for (int step = 128; step > 0; step >>= 1) {
                if (lane < step) {
                    first[lane] = fmaxf(first[lane], first[lane + step]);
                    second[lane] = fmaxf(second[lane], second[lane + step]);
                }
                __syncthreads();
            }
            scale_a = first[0];
            scale_b = second[0];
            __syncthreads();
        }
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (uint64_t i = lane; i < static_cast<uint64_t>(columns); i += 256) {
            float av = a[base + i], bv = b[base + i];
            if constexpr (Cosine) {
                av = scale_a == 0.0f ? 0.0f : av / scale_a;
                bv = scale_b == 0.0f ? 0.0f : bv / scale_b;
                norm_a = fmaf(av, av, norm_a);
                norm_b = fmaf(bv, bv, norm_b);
            }
            dot = fmaf(av, bv, dot);
        }
        first[lane] = dot;
        second[lane] = norm_a;
        third[lane] = norm_b;
        __syncthreads();
        for (int step = 128; step > 0; step >>= 1) {
            if (lane < step) {
                first[lane] += first[lane + step];
                if constexpr (Cosine) {
                    second[lane] += second[lane + step];
                    third[lane] += third[lane + step];
                }
            }
            __syncthreads();
        }
        if (lane == 0) {
            if constexpr (Cosine) {
                const float denominator = sqrtf(second[0]) * sqrtf(third[0]);
                const float value = denominator == 0.0f ? 0.0f : first[0] / denominator;
                output[row] = isnan(value) ? value : fminf(1.0f, fmaxf(-1.0f, value));
            } else {
                output[row] = first[0];
            }
        }
        __syncthreads();
    }
}

inline cudaError_t validate(const float* a, const float* b, float* output, int rows, int columns) {
    if (rows < 0 || columns < 0) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    if (!output || (columns > 0 && (!a || !b || a == output || b == output))
        || static_cast<uint64_t>(rows) * columns > static_cast<uint64_t>(PTRDIFF_MAX) / sizeof(float))
        return cudaErrorInvalidValue;
    return cudaSuccess;
}
} // namespace detail

// Each contiguous input row has `columns` elements; output has `rows` elements.
inline cudaError_t dot(const float* a, const float* b, float* output, int rows,
                       int columns, cudaStream_t stream = nullptr) {
    const cudaError_t status = detail::validate(a, b, output, rows, columns);
    if (status != cudaSuccess || rows == 0) return status;
    detail::reduce_kernel<false><<<rows < 65535 ? rows : 65535, 256, 0, stream>>>(
        a, b, output, rows, columns);
    return cudaGetLastError();
}

// For finite input, either zero norm (including empty rows) gives cosine 0.
inline cudaError_t cosine(const float* a, const float* b, float* output, int rows,
                          int columns, cudaStream_t stream = nullptr) {
    const cudaError_t status = detail::validate(a, b, output, rows, columns);
    if (status != cudaSuccess || rows == 0) return status;
    detail::reduce_kernel<true><<<rows < 65535 ? rows : 65535, 256, 0, stream>>>(
        a, b, output, rows, columns);
    return cudaGetLastError();
}
} // namespace sfVector
