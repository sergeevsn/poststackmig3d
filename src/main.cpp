#include "config_parser.h"
#include "segy_utils.h"
#include "velocity_reader.h"
#include "migration_kernel.h"
#include "progress_bar.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float estimateMaxVelocity(VelocityProvider& velocity_provider,
                         const std::vector<int32_t>& inline_labels,
                         const std::vector<int32_t>& crossline_labels,
                         size_t n_t, double dt) {
    float max_v = 0.0f;
    
    // Try several points to estimate maximum velocity
    std::vector<size_t> test_il_indices = {0};
    if (inline_labels.size() > 1) {
        test_il_indices.push_back(inline_labels.size() / 2);
        test_il_indices.push_back(inline_labels.size() - 1);
    }
    
    std::vector<size_t> test_xl_indices = {0};
    if (crossline_labels.size() > 1) {
        test_xl_indices.push_back(crossline_labels.size() / 2);
        test_xl_indices.push_back(crossline_labels.size() - 1);
    }
    
    for (size_t il_idx : test_il_indices) {
        if (il_idx >= inline_labels.size()) continue;
        
        for (size_t xl_idx : test_xl_indices) {
            if (xl_idx >= crossline_labels.size()) continue;
            
            float vel = velocity_provider.getVelocity(
                inline_labels[il_idx],
                crossline_labels[xl_idx],
                n_t - 1, dt, n_t);
            
            if (vel > max_v) {
                max_v = vel;
            }
        }
    }
    
    // If estimation failed, use a reasonable default value
    if (max_v < 100.0f) {
        max_v = 3000.0f;  // Reasonable default value
    }
    
    return max_v;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "============================================================" << std::endl;
        std::cout << "Post-stack 3D Kirchhoff Migration" << std::endl;
        std::cout << "============================================================" << std::endl;
        
        // 1. Parse configuration
        Config config = ConfigParser::parseConfigFile(argv[1]);
        ConfigParser::validateConfig(config);
        
        std::cout << "Input file: " << config.input_data << std::endl;
        std::cout << "Output file: " << config.output_data << std::endl;
        std::cout << "Inline step: " << config.inline_step << " m" << std::endl;
        std::cout << "Crossline step: " << config.crossline_step << " m" << std::endl;
        std::cout << "Inline padding: " << config.inline_padding << std::endl;
        std::cout << "Crossline padding: " << config.crossline_padding << std::endl;
        std::cout << "Angle aperture: " << config.angle_aperture << " deg" << std::endl;
        std::cout << "Amplitude correction: " << (config.amp_correction ? "true" : "false") << std::endl;
        
        // Parallelization information
#ifdef USE_OPENMP
        int num_procs = omp_get_num_procs();
        int max_threads = omp_get_max_threads();
        const char* omp_num_threads_env = std::getenv("OMP_NUM_THREADS");
        
        std::cout << "Parallelization: OpenMP enabled" << std::endl;
        std::cout << "Available processors: " << num_procs << std::endl;
        std::cout << "OpenMP threads: " << max_threads;
        if (omp_num_threads_env) {
            std::cout << " (set via OMP_NUM_THREADS=" << omp_num_threads_env << ")";
        } else {
            std::cout << " (default, using all processors)";
        }
        std::cout << std::endl;
#else
        std::cout << "Parallelization: disabled (compile with -DENABLE_OPENMP=ON)" << std::endl;
#endif
        std::cout << "============================================================" << std::endl;
        
        // 2. Open input file and create lookup table
        std::cout << "\n[1/6] Opening input file and creating lookup table..." << std::endl;
        SegyFile input_file(config.input_data);
        LookupTableResult lookup_result = input_file.createLookupTable();
        
        size_t n_il = lookup_result.inline_labels.size();
        size_t n_xl = lookup_result.crossline_labels.size();
        size_t n_t = lookup_result.num_samples;
        double dt = lookup_result.dt_seconds;
        
        std::cout << "Input geometry: IL=" << n_il << ", XL=" << n_xl 
                  << ", T=" << n_t << ", dt=" << dt << "s" << std::endl;
        
        // 3. Read velocity
        std::cout << "\n[2/6] Reading velocity..." << std::endl;
        auto velocity_provider = VelocityProviderFactory::create(config.velocity);
        
        // 4. Prepare output labels with padding
        std::cout << "\n[3/6] Preparing output geometry..." << std::endl;
        
        // Expand dimensions with padding
        std::vector<int32_t> output_inline_labels;
        std::vector<int32_t> output_crossline_labels;
        
        // For inline
        if (config.inline_padding > 0 && n_il > 1) {
            int32_t inline_step = lookup_result.inline_labels[1] - lookup_result.inline_labels[0];
            int32_t first_inline = lookup_result.inline_labels[0] - 
                                  config.inline_padding * inline_step;
            int32_t last_inline = lookup_result.inline_labels[n_il - 1] + 
                                 config.inline_padding * inline_step;
            
            for (int32_t il = first_inline; il <= last_inline; il += inline_step) {
                output_inline_labels.push_back(il);
            }
        } else {
            output_inline_labels = lookup_result.inline_labels;
        }
        
        // For crossline
        if (config.crossline_padding > 0 && n_xl > 1) {
            int32_t crossline_step = lookup_result.crossline_labels[1] - 
                                     lookup_result.crossline_labels[0];
            int32_t first_crossline = lookup_result.crossline_labels[0] - 
                                     config.crossline_padding * crossline_step;
            int32_t last_crossline = lookup_result.crossline_labels[n_xl - 1] + 
                                    config.crossline_padding * crossline_step;
            
            for (int32_t xl = first_crossline; xl <= last_crossline; xl += crossline_step) {
                output_crossline_labels.push_back(xl);
            }
        } else {
            output_crossline_labels = lookup_result.crossline_labels;
        }
        
        // Create output file
        SegyFile::createOutputFile(
            config.output_data,
            lookup_result,
            output_inline_labels,
            output_crossline_labels,
            input_file.getTextHeader(),
            input_file.getBinaryHeader(),
            input_file.getDataFormat()
        );
        
        // 5. Calculate aperture parameters
        std::cout << "\n[4/6] Calculating migration parameters..." << std::endl;
        
        float tan_theta = std::tan(config.angle_aperture * M_PI / 180.0);
        float tan_theta_sq = tan_theta * tan_theta;
        
        // Estimate maximum aperture
        float max_v = estimateMaxVelocity(*velocity_provider,
                                         lookup_result.inline_labels,
                                         lookup_result.crossline_labels,
                                         n_t, dt);
        
        double max_t_val = (n_t - 1) * dt;
        float max_aperture_radius_il = (max_v * static_cast<float>(max_t_val) / 2.0f) * std::sqrt(tan_theta_sq);
        float max_aperture_radius_xl = (max_v * static_cast<float>(max_t_val) / 2.0f) * std::sqrt(tan_theta_sq);
        
        int max_aperture_il = static_cast<int>(max_aperture_radius_il / config.inline_step) + 1;
        int max_aperture_xl = static_cast<int>(max_aperture_radius_xl / config.crossline_step) + 1;
        
        std::cout << std::fixed << std::setprecision(0);
        std::cout << "Maximum velocity: " << max_v << " m/s" << std::endl;
        std::cout << "Maximum time: " << std::setprecision(2) << max_t_val << " s" << std::endl;
        std::cout << "Aperture radius: IL=" << std::setprecision(0) << max_aperture_radius_il 
                  << "m, XL=" << max_aperture_radius_xl << "m" << std::endl;
        std::cout << "Buffer size: IL=±" << max_aperture_il 
                  << " lines, XL=±" << max_aperture_xl << " lines" << std::endl;
        std::cout.unsetf(std::ios_base::fixed);
        
        // Prepare time array
        std::vector<float> t0_times(n_t);
        for (size_t i = 0; i < n_t; ++i) {
            t0_times[i] = static_cast<float>(i * dt);
        }
        
        // 6. Open output file for writing
        std::cout << "\n[5/6] Opening output file for writing..." << std::endl;
        SegyOutputFile output_file(
            config.output_data,
            lookup_result,
            output_inline_labels,
            output_crossline_labels,
            input_file.getTextHeader(),
            input_file.getBinaryHeader(),
            input_file.getDataFormat()
        );
        
        // 7. Main migration loop
        std::cout << "\n[6/6] Starting migration loop..." << std::endl;
        size_t n_output_il = output_inline_labels.size();
        size_t n_output_xl = output_crossline_labels.size();
        ProgressBar progress("Migrating", n_output_il);
        
        // Process all output inlines (including padding)
        for (size_t output_il_idx = 0; output_il_idx < n_output_il; ++output_il_idx) {
            int32_t output_inline_label = output_inline_labels[output_il_idx];
            
            // Find corresponding input inline index (if exists)
            auto input_il_it = std::find(lookup_result.inline_labels.begin(),
                                        lookup_result.inline_labels.end(),
                                        output_inline_label);
            
            size_t input_il_idx;
            bool is_padding_il = (input_il_it == lookup_result.inline_labels.end());
            
            if (is_padding_il) {
                // For padding inline, find nearest real inline
                if (output_inline_label < lookup_result.inline_labels[0]) {
                    input_il_idx = 0;
                } else if (output_inline_label > lookup_result.inline_labels[n_il - 1]) {
                    input_il_idx = n_il - 1;
                } else {
                    // Find closest inline
                    input_il_idx = 0;
                    int32_t min_diff = std::abs(output_inline_label - lookup_result.inline_labels[0]);
                    for (size_t i = 1; i < n_il; ++i) {
                        int32_t diff = std::abs(output_inline_label - lookup_result.inline_labels[i]);
                        if (diff < min_diff) {
                            min_diff = diff;
                            input_il_idx = i;
                        }
                    }
                }
            } else {
                input_il_idx = std::distance(lookup_result.inline_labels.begin(), input_il_it);
            }
            
            // Prepare buffer with padding
            std::vector<std::vector<std::vector<float>>> input_buffer;
            std::vector<int32_t> il_pos_indices;
            std::vector<int32_t> xl_pos_indices;
            
            MigrationKernel::prepareInputBufferWithPadding(
                input_file, lookup_result.lookup_table,
                input_il_idx, lookup_result.inline_labels, 
                lookup_result.crossline_labels,
                max_aperture_il, config.inline_padding,
                config.crossline_padding, n_t,
                input_buffer, il_pos_indices, xl_pos_indices
            );
            
            // Prepare velocity slice with padding
            std::vector<std::vector<float>> velocity_slice;
            MigrationKernel::prepareVelocitySliceWithPadding(
                *velocity_provider, input_il_idx,
                lookup_result.inline_labels,
                lookup_result.crossline_labels,
                config.crossline_padding, n_t, dt,
                velocity_slice
            );
            
            // Output slice WITH padding (to capture migrated data in padding areas)
            size_t n_output_xl_padded = n_output_xl;  // Already includes padding
            std::vector<std::vector<float>> output_slice(
                n_output_xl_padded, std::vector<float>(n_t, 0.0f)
            );
            
            // Calculate current inline position in buffer coordinate system
            // Find position of current inline in il_pos_indices
            // il_pos_indices contains original grid indices
            int32_t current_il_pos_in_buffer = -1;
            for (size_t i = 0; i < il_pos_indices.size(); ++i) {
                if (il_pos_indices[i] == static_cast<int32_t>(input_il_idx)) {
                    current_il_pos_in_buffer = static_cast<int32_t>(i);
                    break;
                }
            }
            
            if (current_il_pos_in_buffer < 0) {
                // For padding inlines, find closest position in buffer
                int32_t closest_idx = 0;
                int32_t min_diff = std::abs(static_cast<int32_t>(il_pos_indices[0]) - static_cast<int32_t>(input_il_idx));
                for (size_t i = 1; i < il_pos_indices.size(); ++i) {
                    int32_t diff = std::abs(static_cast<int32_t>(il_pos_indices[i]) - static_cast<int32_t>(input_il_idx));
                    if (diff < min_diff) {
                        min_diff = diff;
                        closest_idx = static_cast<int32_t>(i);
                    }
                }
                current_il_pos_in_buffer = closest_idx;
            }
            
            // Call migration kernel
            // current_xl_start = 0 because output_slice now includes padding
            // and we want to write to all output traces including padding
            MigrationKernel::kirchhoffKernel3DWithPadding(
                input_buffer, velocity_slice, output_slice,
                il_pos_indices, xl_pos_indices,
                current_il_pos_in_buffer,
                0,  // Start from beginning of buffer (buffer already has padding)
                static_cast<float>(dt),
                t0_times,
                tan_theta_sq,
                static_cast<float>(config.inline_step),
                static_cast<float>(config.crossline_step),
                config.amp_correction
            );
            
            // Write all traces for this inline (including padding)
            for (size_t output_xl_idx = 0; output_xl_idx < n_output_xl_padded; ++output_xl_idx) {
                int32_t output_crossline_label = output_crossline_labels[output_xl_idx];
                
                // Try to read header from input file if this trace exists
                std::vector<char> trace_header;
                if (!is_padding_il) {
                    TraceCoords coords;
                    coords.inline_3d = output_inline_label;
                    coords.crossline_3d = output_crossline_label;
                    
                    auto trace_it = lookup_result.lookup_table.find(coords);
                    if (trace_it != lookup_result.lookup_table.end()) {
                        input_file.readTraceHeader(trace_it->second, trace_header);
                    }
                }
                
                // Write trace (including padding traces)
                output_file.writeTrace(
                    output_il_idx, output_xl_idx,
                    output_slice[output_xl_idx],
                    trace_header
                );
            }
            
            progress.update(output_il_idx + 1);
        }
        
        progress.finish();
        
        // Close output file
        output_file.close();
        
        // 8. Completion
        std::cout << "\n[7/7] Migration complete!" << std::endl;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time).count();
        
        std::cout << "\nTotal time: " << elapsed << " seconds" << std::endl;
        std::cout << "Processed: " << n_il << " inlines × " << n_xl 
                  << " crosslines × " << n_t << " samples" << std::endl;
        
        // Close resources
        velocity_provider->close();
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

