#pragma once

#include <vector_types.h> // For float2

// --- Constants ---
constexpr int MAX_ITERS = 22;
constexpr int MAX_DEPTH = MAX_ITERS;
constexpr unsigned int POW2 = 1 << MAX_DEPTH;
constexpr unsigned int SEG_CAP = 3 + 2 * (POW2 - 1);
constexpr unsigned int HYPO_CAP = 1 + 3 * (POW2 - 1);
constexpr unsigned int NODES_CAP = POW2;

// --- Data Structures for GPU ---
struct SimVertex {
    float2 pos;
    uchar4 color;
    float2 tex;
};

struct SimData {
    // Visualization Buffers
    SimVertex* gl_segments;
    SimVertex* gl_hypotenuses;

    // Geometry Buffers
    float2* seg_begin;
    float2* seg_end;
    float2* hyp_begin;
    float2* hyp_end;

    // Node Buffers (Ping-Pong)
    float2* node_R[2];
    float2* node_H0[2];
    float2* node_H1[2];

    // Counts
    int* seg_count;
    int* hyp_count;

    // Base triangle points
    float2* A_pt;
    float2* B_pt;
    float2* C_pt;
};

// --- Kernel Launchers (called from C++) ---

// Allocates and initializes all necessary GPU memory.
void init_memory(SimData& data);

// Frees all allocated GPU memory.
void free_memory(SimData& data);

// Resets counts and generates the base triangle on the GPU.
void reset_and_build_base_launcher(const SimData& data, float angle_deg);

// Performs one level of fractal expansion.
void expand_once_launcher(const SimData& data, int current_buf, int next_buf, int count);

// Copies the final geometry counts from GPU to host.
void get_counts(const SimData& data, int& seg_c, int& hyp_c);

// Copies the base triangle points from GPU to host for marker drawing.
void get_abc_points(const SimData& data, float2& a, float2& b, float2& c);

// Updates the visualization buffers (screen coordinates and colors).
void update_visualization_launcher(const SimData& data, float2 cam_center, float cam_zoom, int2 win_size, int seg_c, int hyp_c);

// Explicitly synchronize the CPU with the GPU.
// Required before accessing managed memory on the host after an asynchronous kernel launch.
void device_synchronize();
