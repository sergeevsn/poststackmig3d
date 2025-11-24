#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <string>
#include <stdexcept>

struct Config {
    std::string input_data;          // Required
    std::string output_data;         // Required
    double inline_step;              // Required, in meters
    double crossline_step;           // Required, in meters
    int inline_padding;              // Default 0
    int crossline_padding;           // Default 0
    std::string velocity;            // Required (path or number as string)
    double angle_aperture;            // Default 30.0 degrees
    bool amp_correction;             // Default true
    int n_threads;                   // Default 0 (use all available threads)
    
    Config();
};

class ConfigParser {
public:
    static Config parseConfigFile(const std::string& config_path);
    static void validateConfig(const Config& config);
    
private:
    static bool parseBool(const std::string& value);
    static double parseDouble(const std::string& value);
    static int parseInt(const std::string& value);
    static std::string trim(const std::string& str);
    static std::string removeQuotes(const std::string& str);
};

#endif // CONFIG_PARSER_H

