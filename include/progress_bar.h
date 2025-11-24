#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <string>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mutex>

class ProgressBar {
public:
    ProgressBar(const std::string& label, size_t total, size_t width = 50);
    void update(size_t current);
    void finish();
    
private:
    std::string label_;
    size_t total_;
    size_t width_;
    size_t last_percent_;
    
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point last_update_time_;
    size_t last_update_count_;
    
    mutable std::mutex mutex_;  // Mutex for thread-safe operations
    
    void printBar(size_t current);
    std::string formatTime(double seconds);
};

#endif // PROGRESS_BAR_H

