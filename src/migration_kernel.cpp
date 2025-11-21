#include "migration_kernel.h"
#include <cmath>
#include <algorithm>
#include <limits>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_SIMD
#include <immintrin.h>
#include <smmintrin.h>  // For _mm_hadd_ps
#endif

void MigrationKernel::kirchhoffKernel3DWithPadding(
    const std::vector<std::vector<std::vector<float>>>& input_buffer,
    const std::vector<std::vector<float>>& velocity_slice,
    std::vector<std::vector<float>>& output_slice,
    const std::vector<int32_t>& il_pos_indices,
    const std::vector<int32_t>& xl_pos_indices,
    int32_t current_il_pos,
    int32_t current_xl_start,
    float dt,
    const std::vector<float>& t0_times,
    float tan_theta_sq,
    float inline_bin_size,
    float crossline_bin_size,
    bool amp_correction) {
    
    size_t n_xl = output_slice.size();
    size_t n_t = (n_xl > 0) ? output_slice[0].size() : 0;
    size_t n_buf_il = il_pos_indices.size();
    size_t n_buf_xl = xl_pos_indices.size();
    
    if (n_xl == 0 || n_t == 0) {
        return;
    }
    
    // Precompute constants
    float inv_dt = 1.0f / dt;
    float inv_crossline_bin_size = 1.0f / crossline_bin_size;
    
    // Precompute inline distances for all positions
    // current_il_pos is now an index in the buffer, not in original grid
    // il_pos_indices contains original grid positions
    std::vector<float> dist_x_sq_cache(n_buf_il);
    int32_t current_il_orig_pos = (current_il_pos >= 0 && current_il_pos < static_cast<int32_t>(il_pos_indices.size())) 
                                   ? il_pos_indices[current_il_pos] : 0;
    
    for (size_t buf_i = 0; buf_i < n_buf_il; ++buf_i) {
        int32_t il_pos_orig = il_pos_indices[buf_i];
        float dist_x = static_cast<float>(il_pos_orig - current_il_orig_pos) * inline_bin_size;
        dist_x_sq_cache[buf_i] = dist_x * dist_x;
    }
    
    // Parallelize over Crossline (output, WITH padding)
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (size_t xl_idx = 0; xl_idx < n_xl; ++xl_idx) {
        // Calculate index in buffer (with padding)
        // output_slice now includes padding, so we map directly to buffer
        int32_t xl_buf_idx = current_xl_start + static_cast<int32_t>(xl_idx);
        
        if (xl_buf_idx < 0 || xl_buf_idx >= static_cast<int32_t>(n_buf_xl)) {
            // Out of buffer bounds - skip
            continue;
        }
        
        // Precompute constants for time and velocity (for all times at once)
        std::vector<float> t0_sq_cache(n_t);
        std::vector<float> v_rms_cache(n_t);
        std::vector<float> inv_v_rms_sq_cache(n_t);
        std::vector<float> four_inv_v_sq_cache(n_t);
        std::vector<float> max_dist_sq_cache(n_t);
        std::vector<int> max_r_xl_bins_cache(n_t);
        std::vector<int> xl_start_buf_cache(n_t);
        std::vector<int> xl_end_buf_cache(n_t);
        std::vector<bool> valid_sample(n_t, false);
        
        for (size_t t_idx = 0; t_idx < n_t; ++t_idx) {
            float t0 = t0_times[t_idx];
            float v_rms = velocity_slice[xl_buf_idx][t_idx];
            
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
            
#ifdef USE_SIMD
            // SIMD-optimized accumulation for batches
            alignas(32) float partial_sums[8] = {0.0f};
            size_t partial_count = 0;
            __m256 sum_vec = _mm256_setzero_ps();
#endif
            
            // Sum over Inline (buffer)
            for (size_t buf_i = 0; buf_i < n_buf_il; ++buf_i) {
                float dist_x_sq = dist_x_sq_cache[buf_i];
                
                if (dist_x_sq > max_dist_sq) {
                    continue;
                }
                
                // Sum over Crossline in buffer
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
                        
                        // Read from buffer
                        float val_low = input_buffer[buf_i][k_buf][idx_low];
                        float val_high = input_buffer[buf_i][k_buf][idx_low + 1];
                        
                        // SIMD-optimized interpolation
                        float val = simdInterpolate(val_low, val_high, frac);
                        
                        // Amplitude correction
                        if (amp_correction) {
                            // Weight coefficient: 1 / (t * v^2)
                            float v_rms_sq = v_rms * v_rms;
                            float weight = 1.0f / (t_hyp * v_rms_sq + 1e-10f);
                            val *= weight;
                        }
                        
#ifdef USE_SIMD
                        // Accumulate in batch for SIMD summation
                        partial_sums[partial_count] = val;
                        partial_count++;
                        
                        // When 8 values accumulated, sum via SIMD
                        if (partial_count >= 8) {
                            __m256 partial_vec = _mm256_load_ps(partial_sums);
                            sum_vec = _mm256_add_ps(sum_vec, partial_vec);
                            partial_count = 0;
                        }
#else
                        migrated_amp += val;
#endif
                    }
                }
            }
            
#ifdef USE_SIMD
            // Process remainder
            if (partial_count > 0) {
                for (size_t i = 0; i < partial_count; ++i) {
                    partial_sums[i] = partial_sums[i];
                }
                // Fill remainder with zeros for alignment
                for (size_t i = partial_count; i < 8; ++i) {
                    partial_sums[i] = 0.0f;
                }
                __m256 partial_vec = _mm256_load_ps(partial_sums);
                sum_vec = _mm256_add_ps(sum_vec, partial_vec);
            }
            
            // Horizontal summation of all 8 elements
            __m128 sum_low = _mm256_extractf128_ps(sum_vec, 0);
            __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            migrated_amp = _mm_cvtss_f32(sum_128);
#endif
            
            output_slice[xl_idx][t_idx] = migrated_amp;
        }
    }
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
    std::vector<std::vector<std::vector<float>>>& input_buffer,
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
    
    size_t n_buf_il = il_end_buf - il_start_buf;
    size_t n_buf_xl = xl_end_buf - xl_start_buf;
    
    // Create buffer
    input_buffer.resize(n_buf_il);
    for (size_t i = 0; i < n_buf_il; ++i) {
        input_buffer[i].resize(n_buf_xl);
        for (size_t j = 0; j < n_buf_xl; ++j) {
            input_buffer[i][j].resize(n_t, 0.0f);
        }
    }
    
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
            
            // Copy to buffer with crossline padding
            for (size_t xl_idx = 0; xl_idx < n_xl && xl_idx < iline_data.size(); ++xl_idx) {
                int xl_buf_idx = crossline_padding + static_cast<int>(xl_idx);
                if (xl_buf_idx >= 0 && xl_buf_idx < static_cast<int>(n_buf_xl)) {
                    input_buffer[buf_il_idx][xl_buf_idx] = std::move(iline_data[xl_idx]);
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
    std::vector<std::vector<float>>& velocity_slice) {
    
    size_t n_xl = crossline_labels.size();
    size_t n_buf_xl = n_xl + 2 * static_cast<size_t>(crossline_padding);
    
    velocity_slice.resize(n_buf_xl);
    for (size_t i = 0; i < n_buf_xl; ++i) {
        velocity_slice[i].resize(n_t, 0.0f);
    }
    
    if (current_il_idx < inline_labels.size()) {
        int32_t inline_label = inline_labels[current_il_idx];
        
        // Try to use optimized method getVelocityIline
        std::vector<std::vector<float>> velocity_iline = 
            velocity_provider.getVelocityIline(inline_label, crossline_labels, n_t, dt);
        
        // Copy to buffer with padding
        size_t xl_start = crossline_padding;
        
        for (size_t xl_idx = 0; xl_idx < n_xl && (xl_start + xl_idx) < n_buf_xl; ++xl_idx) {
            if (xl_idx < velocity_iline.size()) {
                velocity_slice[xl_start + xl_idx] = velocity_iline[xl_idx];
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

