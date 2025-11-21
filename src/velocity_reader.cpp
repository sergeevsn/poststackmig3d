#include "velocity_reader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <set>
#include <limits>
#include <cctype>

// ConstantVelocityProvider
ConstantVelocityProvider::ConstantVelocityProvider(float velocity)
    : velocity_(velocity) {
    if (velocity_ <= 0) {
        throw std::runtime_error("Velocity must be positive, got: " + 
                               std::to_string(velocity_));
    }
}

float ConstantVelocityProvider::getVelocity(int32_t inline_3d, 
                                           int32_t crossline_3d,
                                           size_t time_sample,
                                           double dt, 
                                           size_t /* n_t */) {
    (void)inline_3d;
    (void)crossline_3d;
    (void)time_sample;
    (void)dt;
    return velocity_;
}

// SegyVelocityProvider
SegyVelocityProvider::SegyVelocityProvider(const std::string& segy_path)
    : segy_path_(segy_path)
    , n_t_(0)
    , dt_(0.0) {
    
    std::cout << "Reading velocity from SEG-Y: " << segy_path_ << std::endl;
    
    velocity_file_.reset(new SegyFile(segy_path_));
    LookupTableResult result = velocity_file_->createLookupTable();
    
    lookup_table_ = result.lookup_table;
    inline_labels_ = result.inline_labels;
    crossline_labels_ = result.crossline_labels;
    n_t_ = result.num_samples;
    dt_ = result.dt_seconds;
    
    // Create times array
    times_.resize(n_t_);
    for (size_t i = 0; i < n_t_; ++i) {
        times_[i] = i * dt_;
    }
    
    std::cout << "Velocity grid: IL=" << inline_labels_.size()
              << ", XL=" << crossline_labels_.size()
              << ", T=" << n_t_ << ", dt=" << dt_ << "s" << std::endl;
}

SegyVelocityProvider::~SegyVelocityProvider() {
    close();
}

void SegyVelocityProvider::close() {
    if (velocity_file_) {
        velocity_file_.reset();
    }
}

size_t SegyVelocityProvider::findNearestIndex(const std::vector<int32_t>& sorted_list, 
                                             int32_t value) {
    if (sorted_list.empty()) {
        return 0;
    }
    
    auto it = std::lower_bound(sorted_list.begin(), sorted_list.end(), value);
    
    if (it == sorted_list.begin()) {
        return 0;
    } else if (it == sorted_list.end()) {
        return sorted_list.size() - 1;
    } else {
        // Choose nearest of the two
        size_t idx = std::distance(sorted_list.begin(), it);
        if (std::abs(sorted_list[idx] - value) < std::abs(sorted_list[idx - 1] - value)) {
            return idx;
        } else {
            return idx - 1;
        }
    }
}

float SegyVelocityProvider::interpolateVelocity(int32_t inline_3d, 
                                                int32_t crossline_3d, 
                                                double time) {
    // Find nearest inline and crossline
    size_t il_idx = findNearestIndex(inline_labels_, inline_3d);
    size_t xl_idx = findNearestIndex(crossline_labels_, crossline_3d);
    
    // Take neighborhood for interpolation (2x2 by coordinates)
    std::vector<size_t> il_indices = {
        (il_idx > 0) ? il_idx - 1 : 0,
        (il_idx < inline_labels_.size() - 1) ? il_idx + 1 : inline_labels_.size() - 1
    };
    std::vector<size_t> xl_indices = {
        (xl_idx > 0) ? xl_idx - 1 : 0,
        (xl_idx < crossline_labels_.size() - 1) ? xl_idx + 1 : crossline_labels_.size() - 1
    };
    
    // Collect values for interpolation
    std::vector<float> values;
    std::vector<float> weights;
    
    for (size_t il_i : il_indices) {
        for (size_t xl_i : xl_indices) {
            int32_t il_val = inline_labels_[il_i];
            int32_t xl_val = crossline_labels_[xl_i];
            
            // Read trace
            std::vector<float> trace = velocity_file_->readTraceByCoords(
                lookup_table_, il_val, xl_val, n_t_);
            
            // Interpolate by time
            float vel;
            if (time <= times_[0]) {
                vel = trace[0];
            } else if (time >= times_[n_t_ - 1]) {
                vel = trace[n_t_ - 1];
            } else {
                double idx_float = time / dt_;
                size_t idx_low = static_cast<size_t>(idx_float);
                if (idx_low >= n_t_ - 1) {
                    vel = trace[n_t_ - 1];
                } else {
                    double frac = idx_float - idx_low;
                    vel = trace[idx_low] * (1.0 - frac) + trace[idx_low + 1] * frac;
                }
            }
            
            // Calculate weight (inversely proportional to distance)
            double dist_il = std::abs(static_cast<double>(il_val - inline_3d));
            double dist_xl = std::abs(static_cast<double>(xl_val - crossline_3d));
            float weight = 1.0f / (1.0f + static_cast<float>(dist_il + dist_xl));
            
            values.push_back(vel);
            weights.push_back(weight);
        }
    }
    
    // Weighted average
    if (weights.empty()) {
        return 0.0f;
    }
    
    float sum_val = 0.0f;
    float sum_weight = 0.0f;
    for (size_t i = 0; i < values.size(); ++i) {
        sum_val += values[i] * weights[i];
        sum_weight += weights[i];
    }
    
    return (sum_weight > 0) ? (sum_val / sum_weight) : values[0];
}

float SegyVelocityProvider::getVelocity(int32_t inline_3d, 
                                       int32_t crossline_3d,
                                       size_t time_sample,
                                       double dt, 
                                        size_t /* n_t */) {
    double time = time_sample * dt;
    
    // Check exact coordinate match
    bool exact_il = std::find(inline_labels_.begin(), inline_labels_.end(), inline_3d) 
                    != inline_labels_.end();
    bool exact_xl = std::find(crossline_labels_.begin(), crossline_labels_.end(), crossline_3d)
                    != crossline_labels_.end();
    
    if (exact_il && exact_xl) {
        // Exact coordinate match - read trace
        std::vector<float> trace = velocity_file_->readTraceByCoords(
            lookup_table_, inline_3d, crossline_3d, n_t_);
        
        // Time interpolation
        if (time <= times_[0]) {
            return trace[0];
        } else if (time >= times_[n_t_ - 1]) {
            return trace[n_t_ - 1];
        } else {
            double idx_float = time / dt_;
            size_t idx_low = static_cast<size_t>(idx_float);
            if (idx_low >= n_t_ - 1) {
                return trace[n_t_ - 1];
            }
            double frac = idx_float - idx_low;
            return trace[idx_low] * (1.0 - frac) + trace[idx_low + 1] * frac;
        }
    }
    
    // No exact match - interpolate by coordinates and time
    return interpolateVelocity(inline_3d, crossline_3d, time);
}

std::vector<std::vector<float>> SegyVelocityProvider::getVelocityIline(
    int32_t inline_3d,
    const std::vector<int32_t>& crossline_labels,
    size_t n_t,
    double dt) {
    
    size_t n_xl = crossline_labels.size();
    std::vector<std::vector<float>> velocity_iline(n_xl, std::vector<float>(n_t, 0.0f));
    
    // If exact inline match - read directly
    bool exact_il = std::find(inline_labels_.begin(), inline_labels_.end(), inline_3d) 
                    != inline_labels_.end();
    
    if (exact_il) {
        for (size_t xl_idx = 0; xl_idx < n_xl; ++xl_idx) {
            int32_t crossline = crossline_labels[xl_idx];
            bool exact_xl = std::find(crossline_labels_.begin(), crossline_labels_.end(), crossline)
                           != crossline_labels_.end();
            
            if (exact_xl) {
                std::vector<float> trace = velocity_file_->readTraceByCoords(
                    lookup_table_, inline_3d, crossline, n_t_);
                
                // Interpolate by time if needed
                if (std::abs(dt - dt_) < 1e-6 && n_t == n_t_) {
                    // Times match - use directly
                    for (size_t t_idx = 0; t_idx < n_t && t_idx < trace.size(); ++t_idx) {
                        velocity_iline[xl_idx][t_idx] = trace[t_idx];
                    }
                } else {
                    // Time interpolation needed
                    for (size_t t_idx = 0; t_idx < n_t; ++t_idx) {
                        velocity_iline[xl_idx][t_idx] = getVelocity(
                            inline_3d, crossline, t_idx, dt, n_t);
                    }
                }
            } else {
                // No exact crossline match - interpolate
                for (size_t t_idx = 0; t_idx < n_t; ++t_idx) {
                    double time = t_idx * dt;
                    velocity_iline[xl_idx][t_idx] = interpolateVelocity(
                        inline_3d, crossline, time);
                }
            }
        }
    } else {
        // No exact inline match - interpolate for all crosslines
        for (size_t xl_idx = 0; xl_idx < n_xl; ++xl_idx) {
            int32_t crossline = crossline_labels[xl_idx];
            for (size_t t_idx = 0; t_idx < n_t; ++t_idx) {
                double time = t_idx * dt;
                velocity_iline[xl_idx][t_idx] = interpolateVelocity(
                    inline_3d, crossline, time);
            }
        }
    }
    
    return velocity_iline;
}

// TableVelocityProvider
TableVelocityProvider::TableVelocityProvider(const std::string& table_path) {
    std::cout << "Reading velocity from table: " << table_path << std::endl;
    
    std::ifstream file(table_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open velocity table file: " + table_path);
    }
    
    std::string line;
    int line_num = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        std::istringstream iss(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        int32_t inline_3d, crossline_3d;
        double time;
        float velocity;
        
        if (!(iss >> inline_3d >> crossline_3d >> time >> velocity)) {
            throw std::runtime_error("Invalid format at line " + std::to_string(line_num) +
                                   ": expected 4 columns (INLINE CROSSLINE TIME VEL)");
        }
        
        VelocityPoint point;
        point.inline_3d = inline_3d;
        point.crossline_3d = crossline_3d;
        point.time = time;
        point.velocity = velocity;
        
        points_.push_back(point);
    }
    
    file.close();
    
    if (points_.empty()) {
        throw std::runtime_error("No valid data found in velocity table");
    }
    
    // Find min/max velocity
    min_vel_ = points_[0].velocity;
    max_vel_ = points_[0].velocity;
    for (const auto& point : points_) {
        if (point.velocity < min_vel_) min_vel_ = point.velocity;
        if (point.velocity > max_vel_) max_vel_ = point.velocity;
    }
    
    std::cout << "Loaded " << points_.size() << " velocity points" << std::endl;
    std::cout << "Velocity range: " << min_vel_ << " - " << max_vel_ << " m/s" << std::endl;
}

float TableVelocityProvider::interpolateVelocity(int32_t inline_3d, 
                                                int32_t crossline_3d, 
                                                double time) {
    // Simple interpolation: find nearest points and weighted average
    std::vector<float> values;
    std::vector<float> weights;
    
    for (const auto& point : points_) {
        double dist_il = std::abs(static_cast<double>(point.inline_3d - inline_3d));
        double dist_xl = std::abs(static_cast<double>(point.crossline_3d - crossline_3d));
        double dist_t = std::abs(point.time - time);
        
        // Distance in 4D space
        double dist = std::sqrt(dist_il * dist_il + dist_xl * dist_xl + dist_t * dist_t);
        
        if (dist < 1e-10) {
            // Exact match
            return point.velocity;
        }
        
        float weight = 1.0f / (1.0f + static_cast<float>(dist));
        values.push_back(point.velocity);
        weights.push_back(weight);
    }
    
    // Weighted average
    float sum_val = 0.0f;
    float sum_weight = 0.0f;
    for (size_t i = 0; i < values.size(); ++i) {
        sum_val += values[i] * weights[i];
        sum_weight += weights[i];
    }
    
    float result = (sum_weight > 0) ? (sum_val / sum_weight) : min_vel_;
    
    // Limit to reasonable values
    if (result <= 0) {
        result = min_vel_;
    }
    
    return result;
}

float TableVelocityProvider::getVelocity(int32_t inline_3d, 
                                        int32_t crossline_3d,
                                        size_t time_sample,
                                        double dt, 
                                        size_t /* n_t */) {
    double time = time_sample * dt;
    return interpolateVelocity(inline_3d, crossline_3d, time);
}

// VelocityProviderFactory
bool VelocityProviderFactory::isNumeric(const std::string& str) {
    if (str.empty()) {
        return false;
    }
    
    size_t start = 0;
    if (str[0] == '-' || str[0] == '+') {
        start = 1;
    }
    
    bool has_dot = false;
    for (size_t i = start; i < str.length(); ++i) {
        if (str[i] == '.') {
            if (has_dot) {
                return false;
            }
            has_dot = true;
        } else if (!std::isdigit(str[i])) {
            return false;
        }
    }
    
    return true;
}

bool VelocityProviderFactory::isSegyFile(const std::string& path) {
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower.find(".sgy") != std::string::npos || 
           lower.find(".segy") != std::string::npos;
}

std::unique_ptr<VelocityProvider> VelocityProviderFactory::create(
    const std::string& velocity_param) {
    
    // Check if it's a number
    if (isNumeric(velocity_param)) {
        float vel_value = std::stof(velocity_param);
        return std::unique_ptr<VelocityProvider>(new ConstantVelocityProvider(vel_value));
    }
    
    // Check file extension
    if (isSegyFile(velocity_param)) {
        return std::unique_ptr<VelocityProvider>(new SegyVelocityProvider(velocity_param));
    }
    
    // Otherwise - text file
    return std::unique_ptr<VelocityProvider>(new TableVelocityProvider(velocity_param));
}

// Default implementation for getVelocityIline
std::vector<std::vector<float>> VelocityProvider::getVelocityIline(
    int32_t inline_3d,
    const std::vector<int32_t>& crossline_labels,
    size_t n_t,
    double dt) {
    
    size_t n_xl = crossline_labels.size();
    std::vector<std::vector<float>> velocity_iline(n_xl, std::vector<float>(n_t, 0.0f));
    
    for (size_t xl_idx = 0; xl_idx < n_xl; ++xl_idx) {
        for (size_t t_idx = 0; t_idx < n_t; ++t_idx) {
            velocity_iline[xl_idx][t_idx] = getVelocity(
                inline_3d, crossline_labels[xl_idx], t_idx, dt, n_t);
        }
    }
    
    return velocity_iline;
}

