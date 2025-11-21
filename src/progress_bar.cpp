#include "progress_bar.h"
#include <cmath>
#include <sstream>

ProgressBar::ProgressBar(const std::string& label, size_t total, size_t width)
    : label_(label)
    , total_(total)
    , width_(width)
    , last_percent_(0)
    , start_time_(std::chrono::high_resolution_clock::now())
    , last_update_time_(start_time_)
    , last_update_count_(0)
{
    if (total_ == 0) {
        total_ = 1; // Avoid division by zero
    }
}

std::string ProgressBar::formatTime(double seconds) {
    if (seconds < 0) {
        return "?";
    }
    
    int total_seconds = static_cast<int>(std::round(seconds));
    int hours = total_seconds / 3600;
    int minutes = (total_seconds % 3600) / 60;
    int secs = total_seconds % 60;
    
    std::ostringstream oss;
    oss << std::setfill('0');
    if (hours > 0) {
        oss << hours << ":" << std::setw(2) << minutes 
            << ":" << std::setw(2) << secs;
    } else {
        oss << minutes << ":" << std::setw(2) << secs;
    }
    return oss.str();
}

void ProgressBar::printBar(size_t current) {
    if (total_ == 0) {
        return;
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(now - start_time_).count();
    
    double progress = static_cast<double>(current) / total_;
    size_t percent = static_cast<size_t>(std::round(progress * 100.0));
    
    // Update only when percent changes or every second
    double time_since_last_update = std::chrono::duration<double>(now - last_update_time_).count();
    if (percent == last_percent_ && current < total_ && time_since_last_update < 0.5) {
        return;
    }
    last_percent_ = percent;
    last_update_time_ = now;
    last_update_count_ = current;
    
    size_t bar_width = static_cast<size_t>(std::round(progress * width_));
    
    // Calculate rate and ETA
    double rate = 0.0;
    double eta_seconds = 0.0;
    
    if (elapsed_seconds > 0.0 && current > 0) {
        rate = static_cast<double>(current) / elapsed_seconds;
        
        if (rate > 0.0 && current < total_) {
            size_t remaining = total_ - current;
            eta_seconds = static_cast<double>(remaining) / rate;
        }
    }
    
    std::cout << "\r" << std::left << std::setw(20) << label_ << ": [";
    
    for (size_t i = 0; i < width_; ++i) {
        std::cout << (i < bar_width ? '#' : '.');
    }
    
    std::cout << "] " << std::right << std::setw(3) << percent << "%"
              << " (" << current << "/" << total_ << ")";
    
    // Add time information
    if (elapsed_seconds > 0.0) {
        std::cout << " " << formatTime(elapsed_seconds);
        
        if (eta_seconds > 0.0 && current < total_) {
            std::cout << " < " << formatTime(eta_seconds);
        }
        
        if (rate > 0.0) {
            std::cout << ", " << std::fixed << std::setprecision(1) << rate << " it/s";
            std::cout.unsetf(std::ios_base::fixed);
        }
    }
    
    std::cout << std::flush;
    
    if (current >= total_) {
        std::cout << std::endl;
    }
}

void ProgressBar::update(size_t current) {
    printBar(current);
}

void ProgressBar::finish() {
    printBar(total_);
}

