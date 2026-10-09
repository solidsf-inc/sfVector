// SPDX-License-Identifier: MIT
#include "../src/kernel.cuh"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static void require(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
    float* data = nullptr;
    size_t count;
    explicit Buffer(size_t size) : count(size) { require(cudaMalloc(&data, count * sizeof(float))); }
    ~Buffer() { cudaFree(data); }
    void write(const std::vector<float>& values, cudaStream_t stream) {
        require(cudaMemcpyAsync(data, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    std::vector<float> read(cudaStream_t stream) {
        std::vector<float> result(count);
        require(cudaMemcpyAsync(result.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        require(cudaStreamSynchronize(stream));
        return result;
    }
};

static bool reduction_case(int rows, int columns, bool cosine, bool extremes, cudaStream_t stream) {
    std::vector<float> a(rows * columns), b(rows * columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const int index = row * columns + column;
            a[index] = row == 0 ? 0.0f : (column % 11 - 5) * 0.375f;
            b[index] = row == 1 ? a[index] : row == 2 ? -a[index] : (column % 7 - 3) * 0.625f;
            if (extremes) {
                a[index] *= row % 2 ? 1e30f : 1e-30f;
                b[index] *= row % 2 ? 1e-30f : 1e30f;
            }
        }
    }
    Buffer da(a.size()), db(b.size()), out(rows + 8);
    da.write(a, stream); db.write(b, stream); out.write(std::vector<float>(rows + 8, 633.0f), stream);
    require(cosine ? sfVector::cosine(da.data, db.data, out.data, rows, columns, stream)
                   : sfVector::dot(da.data, db.data, out.data, rows, columns, stream));
    const auto result = out.read(stream);
    for (int row = 0; row < rows; ++row) {
        long double dot = 0.0L, aa = 0.0L, bb = 0.0L;
        for (int column = 0; column < columns; ++column) {
            const long double av = a[row * columns + column], bv = b[row * columns + column];
            dot += av * bv; aa += av * av; bb += bv * bv;
        }
        const long double expected = cosine ? aa == 0.0L || bb == 0.0L ? 0.0L : dot / (std::sqrt(aa) * std::sqrt(bb)) : dot;
        if (!std::isfinite(result[row]) || std::abs(result[row] - expected) > 1e-5L * (1.0L + std::abs(expected)))
            return false;
    }
    for (size_t i = rows; i < result.size(); ++i) if (result[i] != 633.0f) return false;
    return true;
}

static bool empty_rows(bool cosine, cudaStream_t stream) {
    Buffer out(11); out.write(std::vector<float>(11, 633.0f), stream);
    require(cosine ? sfVector::cosine(nullptr, nullptr, out.data, 3, 0, stream)
                   : sfVector::dot(nullptr, nullptr, out.data, 3, 0, stream));
    const auto result = out.read(stream);
    for (size_t i = 0; i < result.size(); ++i) if (result[i] != (i < 3 ? 0.0f : 633.0f)) return false;
    return true;
}

int main() {
    std::vector<std::pair<std::string, bool>> cases;
    cudaStream_t stream = nullptr;
    try {
        cases.emplace_back("invalid_dimensions", sfVector::dot(nullptr, nullptr, nullptr, -1, 5) == cudaErrorInvalidValue
            && sfVector::cosine(nullptr, nullptr, nullptr, 1, -1) == cudaErrorInvalidValue
            && sfVector::dot(nullptr, nullptr, nullptr, 1, 5) == cudaErrorInvalidValue);
        cases.emplace_back("zero_batch", sfVector::dot(nullptr, nullptr, nullptr, 0, 1031) == cudaSuccess
            && sfVector::cosine(nullptr, nullptr, nullptr, 0, 1031) == cudaSuccess);
        require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cases.emplace_back("dot_scalar_rows", reduction_case(7, 1, false, false, stream));
        cases.emplace_back("dot_batched_tail_1031", reduction_case(7, 1031, false, false, stream));
        cases.emplace_back("cosine_zero_parallel_antiparallel", reduction_case(7, 259, true, false, stream));
        cases.emplace_back("cosine_large_and_tiny_magnitudes", reduction_case(7, 1031, true, true, stream));
        cases.emplace_back("dot_empty_rows_zero", empty_rows(false, stream));
        cases.emplace_back("cosine_empty_rows_zero", empty_rows(true, stream));
        Buffer alias(16);
        cases.emplace_back("output_input_alias_rejected", sfVector::dot(alias.data, alias.data, alias.data, 4, 4) == cudaErrorInvalidValue);
        require(cudaStreamDestroy(stream)); stream = nullptr;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        cases.emplace_back("cuda_execution", false);
        if (stream) cudaStreamDestroy(stream);
    }
    bool passed = true;
    for (const auto& entry : cases) passed &= entry.second;
    std::printf("{\"package\":\"sfVector\",\"status\":\"%s\",\"cases\":[", passed ? "PASS" : "FAIL");
    for (size_t i = 0; i < cases.size(); ++i)
        std::printf("%s{\"name\":\"%s\",\"status\":\"%s\"}", i ? "," : "", cases[i].first.c_str(), cases[i].second ? "PASS" : "FAIL");
    std::puts("]}");
    return passed ? 0 : 1;
}
