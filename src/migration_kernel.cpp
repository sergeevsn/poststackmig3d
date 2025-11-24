#include "migration_kernel.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>
#include <cstring>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_SIMD
#include <immintrin.h>
#include <smmintrin.h>  // For _mm_hadd_ps
#endif

void MigrationKernel::kirchhoffKernel3DWithPadding(
    const float* input_buffer,
    size_t n_buf_il, size_t n_buf_xl, size_t n_t,
    const float* velocity_slice,
    std::vector<std::vector<float>>& output_slice,
    const std::vector<int32_t>& il_pos_indices,
    const std::vector<int32_t>& /* xl_pos_indices */,
    int32_t current_il_pos,
    int32_t current_xl_start,
    float dt,
    const std::vector<float>& t0_times,
    float tan_theta_sq,
    float inline_bin_size,
    float crossline_bin_size,
    bool amp_correction) {
    
    size_t n_xl = output_slice.size();
    
    if (n_xl == 0 || n_t == 0) {
        return;
    }
    
    // Precompute strides for flat memory access
    size_t stride_xl = n_buf_xl * n_t;  // Stride for inline dimension
    size_t stride_t = n_t;              // Stride for crossline dimension
    
    // Precompute constants
    float inv_dt = 1.0f / dt;
    float inv_crossline_bin_size = 1.0f / crossline_bin_size;
    
    // Precompute inline distances for all positions
    int32_t current_il_orig_pos = (current_il_pos >= 0 && current_il_pos < static_cast<int32_t>(il_pos_indices.size())) 
                                   ? il_pos_indices[current_il_pos] : 0;
    
    std::vector<float> dist_x_sq_cache(n_buf_il);
    for (size_t buf_i = 0; buf_i < n_buf_il; ++buf_i) {
        int32_t il_pos_orig = il_pos_indices[buf_i];
        float dist_x = static_cast<float>(il_pos_orig - current_il_orig_pos) * inline_bin_size;
        dist_x_sq_cache[buf_i] = dist_x * dist_x;
    }
    
    // Allocate thread-local storage for cache arrays to avoid heap contention
    // Use thread-local storage or allocate once per thread
#ifdef USE_OPENMP
    #pragma omp parallel
    {
        // Thread-local storage for cache arrays
        std::vector<float> t0_sq_cache(n_t);
        std::vector<float> v_rms_cache(n_t);
        std::vector<float> inv_v_rms_sq_cache(n_t);
        std::vector<float> four_inv_v_sq_cache(n_t);
        std::vector<float> max_dist_sq_cache(n_t);
        std::vector<int> max_r_xl_bins_cache(n_t);
        std::vector<int> xl_start_buf_cache(n_t);
        std::vector<int> xl_end_buf_cache(n_t);
        std::vector<bool> valid_sample(n_t, false);
        
        #pragma omp for
        for (size_t xl_idx = 0; xl_idx < n_xl; ++xl_idx) {
            // Calculate index in buffer (with padding)
            int32_t xl_buf_idx = current_xl_start + static_cast<int32_t>(xl_idx);
            
            if (xl_buf_idx < 0 || xl_buf_idx >= static_cast<int32_t>(n_buf_xl)) {
                // Out of buffer bounds - skip
                for (size_t t_idx = 0; t_idx < n_t; ++t_idx) {
                    output_slice[xl_idx][t_idx] = 0.0f;
                }
                continue;
            }
            
            // Precompute constants for time and velocity (for all times at once)
            // Arrays already allocated in thread-local scope
#else
        for (size_t xl_idx = 0; xl_idx < n_xl; ++xl_idx) {
            // Calculate index in buffer (with padding)
            int32_t xl_buf_idx = current_xl_start + static_cast<int32_t>(xl_idx);
            
            if (xl_buf_idx < 0 || xl_buf_idx >= static_cast<int32_t>(n_buf_xl)) {
                // Out of buffer bounds - skip
                for (size_t t_idx = 0; t_idx < n_t; ++t_idx) {
                    output_slice[xl_idx][t_idx] = 0.0f;
                }
                continue;
            }
            
            // Precompute constants for time and velocity (for all times at once)
            // Allocate arrays for non-OpenMP case
            std::vector<float> t0_sq_cache(n_t);
            std::vector<float> v_rms_cache(n_t);
            std::vector<float> inv_v_rms_sq_cache(n_t);
            std::vector<float> four_inv_v_sq_cache(n_t);
            std::vector<float> max_dist_sq_cache(n_t);
            std::vector<int> max_r_xl_bins_cache(n_t);
            std::vector<int> xl_start_buf_cache(n_t);
            std::vector<int> xl_end_buf_cache(n_t);
            std::vector<bool> valid_sample(n_t, false);
#endif
        
        for (size_t t_idx = 0; t_idx < n_t; ++t_idx) {
            float t0 = t0_times[t_idx];
            // Access velocity from flat buffer: velocity[xl_buf_idx * n_t + t_idx]
            float v_rms = velocity_slice[xl_buf_idx * n_t + t_idx];
            
            // Skip too small times or velocities
            if (v_rms >= 1e-5f && t0 >= 1e-5f) {
                valid_sample[t_idx] = true;
                
                // Precompute constants for this time and velocity
                t0_sq_cache[t_idx] = t0 * t0;
                v_rms_cache[t_idx] = v_rms;
                float v_rms_sq = v_rms * v_rms;
                inv_v_rms_sq_cache[t_idx] = 1.0f / (v_rms_sq + 1e-10f);
                four_inv_v_sq_cache[t_idx] = 4.0f * inv_v_rms_sq_cache[t_idx];
                
                // Calculate aperture
                // R_max = (v * t / 2) * tan(theta)
                float v_t_half = v_rms * t0 * 0.5f;
                max_dist_sq_cache[t_idx] = v_t_half * v_t_half * tan_theta_sq;
                
                // Convert meters to bins for crossline
                float max_r = std::sqrt(max_dist_sq_cache[t_idx]);
                max_r_xl_bins_cache[t_idx] = static_cast<int>(max_r * inv_crossline_bin_size) + 1;
                
                // Limit loop bounds for XL in buffer
                xl_start_buf_cache[t_idx] = std::max(0, xl_buf_idx - max_r_xl_bins_cache[t_idx]);
                xl_end_buf_cache[t_idx] = std::min(static_cast<int>(n_buf_xl), xl_buf_idx + max_r_xl_bins_cache[t_idx] + 1);
            }
        }
        
        // Process each time sample
        for (size_t t_idx = 0; t_idx < n_t; ++t_idx) {
            if (!valid_sample[t_idx]) {
                output_slice[xl_idx][t_idx] = 0.0f;
                continue;
            }
            
            float t0_sq = t0_sq_cache[t_idx];
            float v_rms = v_rms_cache[t_idx];
            float four_inv_v_sq = four_inv_v_sq_cache[t_idx];
            float max_dist_sq = max_dist_sq_cache[t_idx];
            int xl_start_buf = xl_start_buf_cache[t_idx];
            int xl_end_buf = xl_end_buf_cache[t_idx];
            
            float migrated_amp = 0.0f;
            
            // Sum over Inline (buffer) - NO nested parallelization
            for (size_t buf_i = 0; buf_i < n_buf_il; ++buf_i) {
                float dist_x_sq = dist_x_sq_cache[buf_i];
                
                if (dist_x_sq > max_dist_sq) {
                    continue;
                }
                
#ifdef USE_SIMD
                // SIMD-optimized processing: process 8 crossline positions at once
                const int simd_width = 8;
                int xl_range = xl_end_buf - xl_start_buf;
                int simd_iterations = xl_range / simd_width;
                
                // Prepare SIMD constants
                __m256 t0_sq_vec = _mm256_set1_ps(t0_sq);
                __m256 four_inv_v_sq_vec = _mm256_set1_ps(four_inv_v_sq);
                __m256 max_dist_sq_vec = _mm256_set1_ps(max_dist_sq);
                __m256 dist_x_sq_vec = _mm256_set1_ps(dist_x_sq);
                __m256 inv_dt_vec = _mm256_set1_ps(inv_dt);
                __m256 crossline_bin_size_vec = _mm256_set1_ps(crossline_bin_size);
                __m256 xl_buf_idx_vec = _mm256_set1_ps(static_cast<float>(xl_buf_idx));
                
                __m256 sum_vec = _mm256_setzero_ps();
                
                // Process 8 crossline positions at once
                for (int simd_i = 0; simd_i < simd_iterations; ++simd_i) {
                    int k_buf_start = xl_start_buf + simd_i * simd_width;
                    
                    // Create indices [k_buf_start, k_buf_start+1, ..., k_buf_start+7]
                    __m256i k_buf_indices = _mm256_setr_epi32(
                        k_buf_start, k_buf_start + 1, k_buf_start + 2, k_buf_start + 3,
                        k_buf_start + 4, k_buf_start + 5, k_buf_start + 6, k_buf_start + 7
                    );
                    
                    // Calculate dist_y for 8 positions: (k_buf - xl_buf_idx) * crossline_bin_size
                    __m256 k_buf_float = _mm256_cvtepi32_ps(k_buf_indices);
                    __m256 dist_y = _mm256_sub_ps(k_buf_float, xl_buf_idx_vec);
                    dist_y = _mm256_mul_ps(dist_y, crossline_bin_size_vec);
                    __m256 dist_y_sq = _mm256_mul_ps(dist_y, dist_y);
                    
                    // Calculate r_sq = dist_x_sq + dist_y_sq
                    __m256 r_sq = _mm256_add_ps(dist_x_sq_vec, dist_y_sq);
                    
                    // Check if r_sq > max_dist_sq (mask)
                    __m256 mask = _mm256_cmp_ps(r_sq, max_dist_sq_vec, _CMP_LE_OQ);
                    
                    // Calculate t_hyp = sqrt(t0^2 + four_inv_v_sq * r_sq) for all 8
                    __m256 four_inv_v_sq_r_sq = _mm256_mul_ps(four_inv_v_sq_vec, r_sq);
                    __m256 t_hyp_sq = _mm256_add_ps(t0_sq_vec, four_inv_v_sq_r_sq);
                    __m256 t_hyp = _mm256_sqrt_ps(t_hyp_sq);
                    
                    // Calculate sample indices: sample_idx = t_hyp * inv_dt
                    __m256 sample_idx_float = _mm256_mul_ps(t_hyp, inv_dt_vec);
                    
                    // Convert to integers (truncate)
                    __m256i idx_low = _mm256_cvttps_epi32(sample_idx_float);
                    
                    // Calculate fractions
                    __m256 idx_low_float = _mm256_cvtepi32_ps(idx_low);
                    __m256 frac = _mm256_sub_ps(sample_idx_float, idx_low_float);
                    
                    // Check bounds: idx_low < n_t - 1 (as float comparison)
                    __m256 n_t_minus_1_float = _mm256_set1_ps(static_cast<float>(n_t) - 1.0f);
                    __m256 bounds_mask = _mm256_cmp_ps(idx_low_float, n_t_minus_1_float, _CMP_LT_OQ);
                    
                    // Combine masks: valid if (r_sq <= max_dist_sq) && (idx_low < n_t - 1)
                    __m256 combined_mask = _mm256_and_ps(mask, bounds_mask);
                    
                    // Store idx_low to array first (fix for _mm256_extract_epi32 requiring compile-time constant)
                    alignas(32) int idx_low_array[8];
                    _mm256_store_si256(reinterpret_cast<__m256i*>(idx_low_array), idx_low);
                    
                    // Gather values from input buffer
                    // Access: input_buffer[buf_i * stride_xl + k_buf * stride_t + idx_low]
                    alignas(32) float values_low[8] = {0.0f};
                    alignas(32) float values_high[8] = {0.0f};
                    
                    for (int lane = 0; lane < simd_width; ++lane) {
                        int k_buf = k_buf_start + lane;
                        int idx_low_val = idx_low_array[lane];
                        int idx_high_val = idx_low_val + 1;
                        
                        // Check bounds and mask
                        if (idx_low_val >= 0 && idx_low_val < static_cast<int>(n_t) - 1) {
                            size_t offset_low = buf_i * stride_xl + k_buf * stride_t + idx_low_val;
                            size_t offset_high = buf_i * stride_xl + k_buf * stride_t + idx_high_val;
                            if (offset_low < (buf_i + 1) * stride_xl && offset_high < (buf_i + 1) * stride_xl) {
                                values_low[lane] = input_buffer[offset_low];
                                values_high[lane] = input_buffer[offset_high];
                            }
                        }
                    }
                    
                    // Load values into SIMD registers
                    __m256 val_low_vec = _mm256_load_ps(values_low);
                    __m256 val_high_vec = _mm256_load_ps(values_high);
                    
                    // Linear interpolation: val = val_low + frac * (val_high - val_low)
                    __m256 diff = _mm256_sub_ps(val_high_vec, val_low_vec);
                    __m256 val = _mm256_fmadd_ps(frac, diff, val_low_vec);
                    
                    // Amplitude correction
                    if (amp_correction) {
                        // Weight = 1 / (t_hyp * v_rms^2)
                        float v_rms_sq = v_rms * v_rms;
                        __m256 v_rms_sq_vec = _mm256_set1_ps(v_rms_sq);
                        __m256 t_hyp_v_sq = _mm256_mul_ps(t_hyp, v_rms_sq_vec);
                        __m256 weight = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(t_hyp_v_sq, _mm256_set1_ps(1e-10f)));
                        val = _mm256_mul_ps(val, weight);
                    }
                    
                    // Apply mask (zero out invalid lanes) and accumulate
                    val = _mm256_and_ps(val, combined_mask);
                    sum_vec = _mm256_add_ps(sum_vec, val);
                }
                
                // Process remainder (scalar)
                for (int k_buf = xl_start_buf + simd_iterations * simd_width; k_buf < xl_end_buf; ++k_buf) {
                    // Crossline distance (in meters)
                    float dist_y = static_cast<float>(k_buf - xl_buf_idx) * crossline_bin_size;
                    float dist_y_sq = dist_y * dist_y;
                    float r_sq = dist_x_sq + dist_y_sq;
                    
                    if (r_sq > max_dist_sq) {
                        continue;
                    }
                    
                    // Hyperbolic time: t_hyp = sqrt(t0^2 + 4*r^2/v^2)
                    float t_hyp = std::sqrt(t0_sq + four_inv_v_sq * r_sq);
                    
                    // Calculate index (sample)
                    float sample_idx_float = t_hyp * inv_dt;
                    int idx_low = static_cast<int>(sample_idx_float);
                    
                    // Check bounds
                    if (idx_low < static_cast<int>(n_t) - 1) {
                        float frac = sample_idx_float - idx_low;
                        
                        // Read from flat buffer: input_buffer[buf_i * stride_xl + k_buf * stride_t + idx_low]
                        size_t offset_low = buf_i * stride_xl + k_buf * stride_t + idx_low;
                        size_t offset_high = buf_i * stride_xl + k_buf * stride_t + idx_low + 1;
                        
                        float val_low = input_buffer[offset_low];
                        float val_high = input_buffer[offset_high];
                        
                        // Linear interpolation
                        float val = simdInterpolate(val_low, val_high, frac);
                        
                        // Amplitude correction
                        if (amp_correction) {
                            float v_rms_sq = v_rms * v_rms;
                            float weight = 1.0f / (t_hyp * v_rms_sq + 1e-10f);
                            val *= weight;
                        }
                        
                        migrated_amp += val;
                    }
                }
                
                // Horizontal summation of SIMD accumulator
                alignas(32) float sum_array[8];
                _mm256_store_ps(sum_array, sum_vec);
                for (int i = 0; i < 8; ++i) {
                    migrated_amp += sum_array[i];
                }
                
#else
                // Scalar version (no SIMD)
                for (int k_buf = xl_start_buf; k_buf < xl_end_buf; ++k_buf) {
                    // Crossline distance (in meters)
                    float dist_y = static_cast<float>(k_buf - xl_buf_idx) * crossline_bin_size;
                    float dist_y_sq = dist_y * dist_y;
                    float r_sq = dist_x_sq + dist_y_sq;
                    
                    if (r_sq > max_dist_sq) {
                        continue;
                    }
                    
                    // Hyperbolic time: t_hyp = sqrt(t0^2 + 4*r^2/v^2)
                    float t_hyp = std::sqrt(t0_sq + four_inv_v_sq * r_sq);
                    
                    // Calculate index (sample)
                    float sample_idx_float = t_hyp * inv_dt;
                    int idx_low = static_cast<int>(sample_idx_float);
                    
                    // Check bounds
                    if (idx_low < static_cast<int>(n_t) - 1) {
                        float frac = sample_idx_float - idx_low;
                        
                        // Read from flat buffer: input_buffer[buf_i * stride_xl + k_buf * stride_t + idx_low]
                        size_t offset_low = buf_i * stride_xl + k_buf * stride_t + idx_low;
                        size_t offset_high = buf_i * stride_xl + k_buf * stride_t + idx_low + 1;
                        
                        float val_low = input_buffer[offset_low];
                        float val_high = input_buffer[offset_high];
                        
                        // Linear interpolation
                        float val = simdInterpolate(val_low, val_high, frac);
                        
                        // Amplitude correction
                        if (amp_correction) {
                            float v_rms_sq = v_rms * v_rms;
                            float weight = 1.0f / (t_hyp * v_rms_sq + 1e-10f);
                            val *= weight;
                        }
                        
                        migrated_amp += val;
                    }
                }
#endif
            }
            
            output_slice[xl_idx][t_idx] = migrated_amp;
        }
#ifdef USE_OPENMP
        }  // End of for loop
    }  // End of parallel region
#else
        }  // End of for loop
#endif
}

void MigrationKernel::prepareInputBufferWithPadding(
    SegyFile& segy_file,
    const LookupTable& lookup_table,
    size_t current_il_idx,
    const std::vector<int32_t>& inline_labels,
    const std::vector<int32_t>& crossline_labels,
    int max_aperture_il,
    int inline_padding,
    int crossline_padding,
    size_t n_t,
    std::vector<float>& input_buffer,
    size_t& n_buf_il, size_t& n_buf_xl,
    std::vector<int32_t>& il_pos_indices,
    std::vector<int32_t>& xl_pos_indices) {
    
    size_t n_il = inline_labels.size();
    size_t n_xl = crossline_labels.size();
    
    // Determine inline range for buffer (with aperture)
    int il_start_req = std::max(0, static_cast<int>(current_il_idx) - max_aperture_il);
    int il_end_req = std::min(static_cast<int>(n_il), static_cast<int>(current_il_idx) + max_aperture_il + 1);
    
    // Add inline padding
    int il_start_buf = std::max(0, il_start_req - inline_padding);
    int il_end_buf = std::min(static_cast<int>(n_il), il_end_req + inline_padding);
    
    // Add crossline padding
    int xl_start_buf = 0;  // Always start at 0 for buffer
    int xl_end_buf = static_cast<int>(n_xl) + 2 * crossline_padding;
    
    n_buf_il = il_end_buf - il_start_buf;
    n_buf_xl = xl_end_buf - xl_start_buf;
    
    // Create flat buffer: n_buf_il * n_buf_xl * n_t
    size_t buffer_size = n_buf_il * n_buf_xl * n_t;
    input_buffer.resize(buffer_size, 0.0f);
    
    // Precompute strides
    size_t stride_xl = n_buf_xl * n_t;
    size_t stride_t = n_t;
    
    // Positional indices for inline (in original coordinate system)
    il_pos_indices.clear();
    il_pos_indices.reserve(n_buf_il);
    for (int i = il_start_buf; i < il_end_buf; ++i) {
        il_pos_indices.push_back(static_cast<int32_t>(i));
    }
    
    // Positional indices for crossline in buffer
    xl_pos_indices.clear();
    xl_pos_indices.reserve(n_buf_xl);
    for (int i = 0; i < xl_end_buf; ++i) {
        xl_pos_indices.push_back(static_cast<int32_t>(i));
    }
    
    // Fill buffer with data - optimized reading of entire inlines
    for (size_t buf_il_idx = 0; buf_il_idx < n_buf_il; ++buf_il_idx) {
        int il_pos = il_start_buf + static_cast<int>(buf_il_idx);
        
        if (il_pos >= 0 && il_pos < static_cast<int>(n_il)) {
            // Real data
            int32_t inline_label = inline_labels[il_pos];
            
            // Optimization: read entire inline at once instead of traces one by one
            std::vector<std::vector<float>> iline_data = segy_file.readIlineByCoords(
                lookup_table, inline_label, crossline_labels, n_t);
            
            // Copy to flat buffer with crossline padding
            for (size_t xl_idx = 0; xl_idx < n_xl && xl_idx < iline_data.size(); ++xl_idx) {
                int xl_buf_idx = crossline_padding + static_cast<int>(xl_idx);
                if (xl_buf_idx >= 0 && xl_buf_idx < static_cast<int>(n_buf_xl)) {
                    // Copy trace to flat buffer: buffer[buf_il_idx * stride_xl + xl_buf_idx * stride_t + t]
                    size_t base_offset = buf_il_idx * stride_xl + xl_buf_idx * stride_t;
                    const std::vector<float>& trace = iline_data[xl_idx];
                    size_t copy_size = std::min(n_t, trace.size());
                    if (copy_size > 0) {
                        std::memcpy(&input_buffer[base_offset], trace.data(), copy_size * sizeof(float));
                    }
                }
            }
        }
        // Otherwise remains zero (padding)
    }
}

void MigrationKernel::prepareVelocitySliceWithPadding(
    VelocityProvider& velocity_provider,
    size_t current_il_idx,
    const std::vector<int32_t>& inline_labels,
    const std::vector<int32_t>& crossline_labels,
    int crossline_padding,
    size_t n_t,
    double dt,
    std::vector<float>& velocity_slice,
    size_t& n_buf_xl) {
    
    size_t n_xl = crossline_labels.size();
    n_buf_xl = n_xl + 2 * static_cast<size_t>(crossline_padding);
    
    // Create flat buffer: n_buf_xl * n_t
    size_t buffer_size = n_buf_xl * n_t;
    velocity_slice.resize(buffer_size, 0.0f);
    
    if (current_il_idx < inline_labels.size()) {
        int32_t inline_label = inline_labels[current_il_idx];
        
        // Try to use optimized method getVelocityIline
        std::vector<std::vector<float>> velocity_iline = 
            velocity_provider.getVelocityIline(inline_label, crossline_labels, n_t, dt);
        
        // Copy to flat buffer with padding
        size_t xl_start = crossline_padding;
        
        for (size_t xl_idx = 0; xl_idx < n_xl && (xl_start + xl_idx) < n_buf_xl; ++xl_idx) {
            if (xl_idx < velocity_iline.size()) {
                // Copy trace to flat buffer: velocity[(xl_start + xl_idx) * n_t + t]
                size_t base_offset = (xl_start + xl_idx) * n_t;
                const std::vector<float>& trace = velocity_iline[xl_idx];
                size_t copy_size = std::min(n_t, trace.size());
                if (copy_size > 0) {
                    std::memcpy(&velocity_slice[base_offset], trace.data(), copy_size * sizeof(float));
                }
            }
        }
    }
    // Padding remains zero
}

#ifdef USE_SIMD
float MigrationKernel::simdSum(const float* data, size_t n) {
    if (n == 0) {
        return 0.0f;
    }
    
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    
    // Process 8 elements at a time
    for (; i + 8 <= n; i += 8) {
        __m256 data_vec = _mm256_loadu_ps(&data[i]);
        sum_vec = _mm256_add_ps(sum_vec, data_vec);
    }
    
    // Horizontal summation
    float sum = 0.0f;
    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    for (int j = 0; j < 8; ++j) {
        sum += temp[j];
    }
    
    // Remainder
    for (; i < n; ++i) {
        sum += data[i];
    }
    
    return sum;
}

float MigrationKernel::simdInterpolate(float val_low, float val_high, float frac) {
    // FMA (Fused Multiply-Add) for more accurate and faster interpolation
    // val = val_low * (1 - frac) + val_high * frac
    //     = val_low + frac * (val_high - val_low)
    float diff = val_high - val_low;
    return std::fma(frac, diff, val_low);
}

void MigrationKernel::simdAccumulate(float& sum, float val) {
    // Simple addition, compiler can optimize this better with -ffast-math
    sum += val;
}
#else
float MigrationKernel::simdSum(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

float MigrationKernel::simdInterpolate(float val_low, float val_high, float frac) {
    float one_minus_frac = 1.0f - frac;
    return val_low * one_minus_frac + val_high * frac;
}

void MigrationKernel::simdAccumulate(float& sum, float val) {
    sum += val;
}
#endif
