#ifndef VELOCITY_READER_H
#define VELOCITY_READER_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include "segy_utils.h"

// Abstract velocity provider class
class VelocityProvider {
public:
    virtual ~VelocityProvider() = default;
    
    // Get velocity for a single point
    virtual float getVelocity(int32_t inline_3d, 
                             int32_t crossline_3d,
                             size_t time_sample,
                             double dt, 
                             size_t n_t) = 0;
    
    // Get entire inline of velocity (optimization)
    virtual std::vector<std::vector<float>> getVelocityIline(
        int32_t inline_3d,
        const std::vector<int32_t>& crossline_labels,
        size_t n_t,
        double dt);
    
    // Close resources (if needed)
    virtual void close() {}
};

// Constant velocity
class ConstantVelocityProvider : public VelocityProvider {
private:
    float velocity_;
    
public:
    explicit ConstantVelocityProvider(float velocity);
    float getVelocity(int32_t inline_3d, 
                     int32_t crossline_3d,
                     size_t time_sample,
                     double dt, 
                     size_t n_t) override;
};

// Velocity from SEG-Y file
class SegyVelocityProvider : public VelocityProvider {
private:
    std::string segy_path_;
    std::unique_ptr<SegyFile> velocity_file_;
    LookupTable lookup_table_;
    std::vector<int32_t> inline_labels_;
    std::vector<int32_t> crossline_labels_;
    size_t n_t_;
    double dt_;
    std::vector<double> times_;
    
public:
    explicit SegyVelocityProvider(const std::string& segy_path);
    ~SegyVelocityProvider();
    
    float getVelocity(int32_t inline_3d, 
                     int32_t crossline_3d,
                     size_t time_sample,
                     double dt, 
                     size_t n_t) override;
    
    std::vector<std::vector<float>> getVelocityIline(
        int32_t inline_3d,
        const std::vector<int32_t>& crossline_labels,
        size_t n_t,
        double dt) override;
    
    void close() override;
    
private:
    float interpolateVelocity(int32_t inline_3d, 
                             int32_t crossline_3d, 
                             double time);
    size_t findNearestIndex(const std::vector<int32_t>& sorted_list, 
                           int32_t value);
};

// Velocity from text table
class TableVelocityProvider : public VelocityProvider {
private:
    struct VelocityPoint {
        int32_t inline_3d;
        int32_t crossline_3d;
        double time;
        float velocity;
    };
    
    std::vector<VelocityPoint> points_;
    float min_vel_;
    float max_vel_;
    
public:
    explicit TableVelocityProvider(const std::string& table_path);
    
    float getVelocity(int32_t inline_3d, 
                     int32_t crossline_3d,
                     size_t time_sample,
                     double dt, 
                     size_t n_t) override;
    
private:
    float interpolateVelocity(int32_t inline_3d, 
                             int32_t crossline_3d, 
                             double time);
};

// Factory for creating providers
class VelocityProviderFactory {
public:
    static std::unique_ptr<VelocityProvider> create(const std::string& velocity_param);
    
private:
    static bool isNumeric(const std::string& str);
    static bool isSegyFile(const std::string& path);
};

#endif // VELOCITY_READER_H

