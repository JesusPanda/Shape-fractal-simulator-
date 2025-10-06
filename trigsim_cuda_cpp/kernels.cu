#include "kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// --- Error Handling ---
#define CUDA_CHECK(err) { \
    cudaError_t e = err; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
}

// --- Device-side Math Helper ---
__device__ inline float2 operator-(const float2& a, const float2& b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ inline float2 operator+(const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ inline float2 operator*(const float2& a, float s) { return make_float2(a.x * s, a.y * s); }
__device__ inline float dot(const float2& a, const float2& b) { return a.x * b.x + a.y * b.y; }

__device__ float2 foot_of_perp(const float2& P, const float2& A, const float2& B) {
    float2 AB = B - A;
    float denom = dot(AB, AB);
    float t = 0.0f;
    if (denom != 0.0f) {
        t = dot(P - A, AB) / denom;
    }
    return A + AB * t;
}

// --- Kernels ---

__global__ void reset_kernel(SimData data) {
    *data.seg_count = 0;
    *data.hyp_count = 0;
}

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

__global__ void build_base_kernel(SimData data, float angle_deg) {
    // Clamp angle
    float a = fminf(85.0f, fmaxf(5.0f, angle_deg));
    float ang = a * M_PI_F / 180.0f;

    // Construct base triangle
    float2 B = make_float2(0.15f, 0.15f);
    float AB_len = 0.70f;
    float BC_len = AB_len * tanf(ang);

    if (BC_len > 0.70f) {
        float s = 0.70f / BC_len;
        AB_len *= s;
        BC_len *= s;
    }

    float2 A = make_float2(B.x + AB_len, B.y);
    float2 C = make_float2(B.x, B.y + BC_len);

    // Persist points
    *data.A_pt = A;
    *data.B_pt = B;
    *data.C_pt = C;

    // Legs
    int s_idx = atomicAdd(data.seg_count, 2);
    data.seg_begin[s_idx] = A; data.seg_end[s_idx] = B;
    data.seg_begin[s_idx+1] = B; data.seg_end[s_idx+1] = C;

    // Altitude
    float2 K = foot_of_perp(B, A, C);
    s_idx = atomicAdd(data.seg_count, 1);
    data.seg_begin[s_idx] = B; data.seg_end[s_idx] = K;

    // Hypotenuse
    int h_idx = atomicAdd(data.hyp_count, 1);
    data.hyp_begin[h_idx] = A; data.hyp_end[h_idx] = C;

    // Init first node
    data.node_R[0][0] = B;
    data.node_H0[0][0] = A;
    data.node_H1[0][0] = C;
}

__global__ void expand_kernel(SimData data, int current_buf, int next_buf, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float2 R  = data.node_R[current_buf][i];
    float2 H0 = data.node_H0[current_buf][i];
    float2 H1 = data.node_H1[current_buf][i];

    float2 F = foot_of_perp(R, H0, H1);
    float2 v = R - F;
    float2 H0p = H0 + v;
    float2 H1p = H1 + v;

    // Connectors
    int s_idx = atomicAdd(data.seg_count, 2);
    data.seg_begin[s_idx] = H0; data.seg_end[s_idx] = H0p;
    data.seg_begin[s_idx+1] = H1; data.seg_end[s_idx+1] = H1p;

    // Hypotenuses
    int h_idx = atomicAdd(data.hyp_count, 3);
    data.hyp_begin[h_idx] = H0p; data.hyp_end[h_idx] = H1p;
    data.hyp_begin[h_idx+1] = R; data.hyp_end[h_idx+1] = H0;
    data.hyp_begin[h_idx+2] = R; data.hyp_end[h_idx+2] = H1;

    // Write next-level nodes
    int j = 2 * i;
    data.node_R[next_buf][j] = H0p;
    data.node_H0[next_buf][j] = R;
    data.node_H1[next_buf][j] = H0;
    data.node_R[next_buf][j+1] = H1p;
    data.node_H0[next_buf][j+1] = R;
    data.node_H1[next_buf][j+1] = H1;
}


// --- Memory Management ---
void init_memory(SimData& data) {
    CUDA_CHECK(cudaMallocManaged(&data.seg_begin, SEG_CAP * sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged(&data.seg_end, SEG_CAP * sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged(&data.hyp_begin, HYPO_CAP * sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged(&data.hyp_end, HYPO_CAP * sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged(&data.node_R[0], NODES_CAP * sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged(&data.node_H0[0], NODES_CAP * sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged(&data.node_H1[0], NODES_CAP * sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged(&data.node_R[1], NODES_CAP * sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged(&data.node_H0[1], NODES_CAP * sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged(&data.node_H1[1], NODES_CAP * sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged(&data.seg_count, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&data.hyp_count, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&data.A_pt, sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged(&data.B_pt, sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged(&data.C_pt, sizeof(float2)));
}

void free_memory(SimData& data) {
    CUDA_CHECK(cudaFree(data.seg_begin));
    CUDA_CHECK(cudaFree(data.seg_end));
    CUDA_CHECK(cudaFree(data.hyp_begin));
    CUDA_CHECK(cudaFree(data.hyp_end));
    CUDA_CHECK(cudaFree(data.node_R[0]));
    CUDA_CHECK(cudaFree(data.node_H0[0]));
    CUDA_CHECK(cudaFree(data.node_H1[0]));
    CUDA_CHECK(cudaFree(data.node_R[1]));
    CUDA_CHECK(cudaFree(data.node_H0[1]));
    CUDA_CHECK(cudaFree(data.node_H1[1]));
    CUDA_CHECK(cudaFree(data.seg_count));
    CUDA_CHECK(cudaFree(data.hyp_count));
    CUDA_CHECK(cudaFree(data.A_pt));
    CUDA_CHECK(cudaFree(data.B_pt));
    CUDA_CHECK(cudaFree(data.C_pt));
}

// --- Kernel Launchers ---
void reset_and_build_base_launcher(const SimData& data, float angle_deg) {
    reset_kernel<<<1, 1>>>(data);
    CUDA_CHECK(cudaDeviceSynchronize());
    build_base_kernel<<<1, 1>>>(data, angle_deg);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void expand_once_launcher(const SimData& data, int current_buf, int next_buf, int count) {
    if (count == 0) return;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    expand_kernel<<<blocks, threads>>>(data, current_buf, next_buf, count);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void get_counts(const SimData& data, int& seg_c, int& hyp_c) {
    seg_c = *data.seg_count;
    hyp_c = *data.hyp_count;
}

void get_abc_points(const SimData& data, float2& a, float2& b, float2& c) {
    a = *data.A_pt;
    b = *data.B_pt;
    c = *data.C_pt;
}