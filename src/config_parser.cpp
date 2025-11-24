#include "config_parser.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <vector>

Config::Config()
    : input_data("")
    , output_data("")
    , inline_step(0.0)
    , crossline_step(0.0)
    , inline_padding(0)
    , crossline_padding(0)
    , velocity("")
    , angle_aperture(30.0)
    , amp_correction(true)
    , n_threads(0)
{
}

std::string ConfigParser::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

std::string ConfigParser::removeQuotes(const std::string& str) {
    std::string result = trim(str);
    if (result.length() >= 2) {
        if ((result.front() == '"' && result.back() == '"') ||
            (result.front() == '\'' && result.back() == '\'')) {
            return result.substr(1, result.length() - 2);
        }
    }
    return result;
}

bool ConfigParser::parseBool(const std::string& value) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    lower = trim(lower);
    
    if (lower == "true" || lower == "1" || lower == "yes" || lower == "on") {
        return true;
    } else if (lower == "false" || lower == "0" || lower == "no" || lower == "off") {
        return false;
    } else {
        throw std::runtime_error("Cannot parse boolean value: " + value);
    }
}

double ConfigParser::parseDouble(const std::string& value) {
    try {
        return std::stod(value);
    } catch (const std::exception& e) {
        throw std::runtime_error("Cannot parse double value: " + value);
    }
}

int ConfigParser::parseInt(const std::string& value) {
    try {
        return std::stoi(value);
    } catch (const std::exception& e) {
        throw std::runtime_error("Cannot parse int value: " + value);
    }
}

Config ConfigParser::parseConfigFile(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Config file not found: " + config_path);
    }
    
    std::map<std::string, std::string> config_dict;
    std::string line;
    int line_num = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Parse key=value
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            throw std::runtime_error("Invalid format at line " + 
                                   std::to_string(line_num) + 
                                   ": expected key=value, got: " + line);
        }
        
        std::string key = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos + 1));
        value = removeQuotes(value);
        
        config_dict[key] = value;
    }
    
    file.close();
    
    // Check required parameters
    std::vector<std::string> required_params = {
        "input_data", "output_data", "inline_step", 
        "crossline_step", "velocity"
    };
    
    std::vector<std::string> missing_params;
    for (const auto& param : required_params) {
        if (config_dict.find(param) == config_dict.end()) {
            missing_params.push_back(param);
        }
    }
    
    if (!missing_params.empty()) {
        std::string msg = "Missing required parameters: ";
        for (size_t i = 0; i < missing_params.size(); ++i) {
            msg += missing_params[i];
            if (i < missing_params.size() - 1) {
                msg += ", ";
            }
        }
        throw std::runtime_error(msg);
    }
    
    // Create configuration
    Config config;
    
    try {
        config.input_data = config_dict["input_data"];
        config.output_data = config_dict["output_data"];
        config.inline_step = parseDouble(config_dict["inline_step"]);
        config.crossline_step = parseDouble(config_dict["crossline_step"]);
        config.velocity = config_dict["velocity"];
        
        // Optional parameters
        if (config_dict.find("inline_padding") != config_dict.end()) {
            config.inline_padding = parseInt(config_dict["inline_padding"]);
        }
        if (config_dict.find("crossline_padding") != config_dict.end()) {
            config.crossline_padding = parseInt(config_dict["crossline_padding"]);
        }
        if (config_dict.find("angle_aperture") != config_dict.end()) {
            config.angle_aperture = parseDouble(config_dict["angle_aperture"]);
        }
        if (config_dict.find("amp_correction") != config_dict.end()) {
            config.amp_correction = parseBool(config_dict["amp_correction"]);
        }
        if (config_dict.find("n_threads") != config_dict.end()) {
            config.n_threads = parseInt(config_dict["n_threads"]);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Error parsing config values: " + 
                               std::string(e.what()));
    }
    
    return config;
}

void ConfigParser::validateConfig(const Config& config) {
    // Check steps
    if (config.inline_step <= 0) {
        throw std::runtime_error("inline_step must be positive, got: " + 
                               std::to_string(config.inline_step));
    }
    if (config.crossline_step <= 0) {
        throw std::runtime_error("crossline_step must be positive, got: " + 
                               std::to_string(config.crossline_step));
    }
    
    // Check padding
    if (config.inline_padding < 0) {
        throw std::runtime_error("inline_padding must be non-negative, got: " + 
                               std::to_string(config.inline_padding));
    }
    if (config.crossline_padding < 0) {
        throw std::runtime_error("crossline_padding must be non-negative, got: " + 
                               std::to_string(config.crossline_padding));
    }
    
    // Check aperture angle
    if (config.angle_aperture <= 0 || config.angle_aperture >= 90) {
        throw std::runtime_error("angle_aperture must be in range (0, 90), got: " + 
                               std::to_string(config.angle_aperture));
    }
    
    // Check n_threads
    if (config.n_threads < 0) {
        throw std::runtime_error("n_threads must be non-negative (0 = use all available), got: " + 
                               std::to_string(config.n_threads));
    }
    
    // Check velocity
    // If it's a number - check that it's positive
    try {
        double vel_value = std::stod(config.velocity);
        if (vel_value <= 0) {
            throw std::runtime_error("Velocity constant must be positive, got: " + 
                                   std::to_string(vel_value));
        }
    } catch (const std::exception&) {
        // Not a number, so it's a file path - check existence later
    }
}

