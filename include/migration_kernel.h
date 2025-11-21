#ifndef MIGRATION_KERNEL_H
#define MIGRATION_KERNEL_H

#include <vector>
#include <cstdint>
#include "segy_utils.h"
#include "velocity_reader.h"

// 3D Kirchhoff migration kernel with padding
// Parallelization via OpenMP, optimization via SIMD
class MigrationKernel {
public:
    // Main migration function
    static void kirchhoffKernel3DWithPadding(
        const std::vector<std::vector<std::vector<float>>>& input_buffer,  // [n_buf_il][n_buf_xl][n_t]
        const std::vector<std::vector<float>>& velocity_slice,              // [n_buf_xl][n_t]
        std::vector<std::vector<float>>& output_slice,                     // [n_xl][n_t] WITH padding
        const std::vector<int32_t>& il_pos_indices,                         // Positional indices for inline
        const std::vector<int32_t>& xl_pos_indices,                        // Positional indices for crossline
        int32_t current_il_pos,                                             // Current positional index for inline
        int32_t current_xl_start,                                           // Start of crossline data in buffer
        float dt,                                                            // Time step
        const std::vector<float>& t0_times,                                 // Time array
        float tan_theta_sq,                                                  // tan^2 of aperture angle
        float inline_bin_size,                                               // Inline step in meters
        float crossline_bin_size,                                            // Crossline step in meters
        bool amp_correction                                                  // Amplitude correction flag
    );
    
    // Prepare input buffer with padding
    static void prepareInputBufferWithPadding(
        SegyFile& segy_file,
        const LookupTable& lookup_table,
        size_t current_il_idx,
        const std::vector<int32_t>& inline_labels,
        const std::vector<int32_t>& crossline_labels,
        int max_aperture_il,
        int inline_padding,
        int crossline_padding,
        size_t n_t,
        std::vector<std::vector<std::vector<float>>>& input_buffer,
        std::vector<int32_t>& il_pos_indices,
        std::vector<int32_t>& xl_pos_indices
    );
    
    // Prepare velocity slice with padding
    static void prepareVelocitySliceWithPadding(
        VelocityProvider& velocity_provider,
        size_t current_il_idx,
        const std::vector<int32_t>& inline_labels,
        const std::vector<int32_t>& crossline_labels,
        int crossline_padding,
        size_t n_t,
        double dt,
        std::vector<std::vector<float>>& velocity_slice
    );
    
private:
    // SIMD-optimized functions (if available)
    static float simdSum(const float* data, size_t n);
    
    // SIMD-optimized linear interpolation
    static float simdInterpolate(float val_low, float val_high, float frac);
    
    // SIMD-optimized sum accumulation
    static void simdAccumulate(float& sum, float val);
};

#endif // MIGRATION_KERNEL_H

