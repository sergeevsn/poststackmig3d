#ifndef SEGY_UTILS_H
#define SEGY_UTILS_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>
#include <memory>

// Coordinate pair (inline, crossline)
struct TraceCoords {
    int32_t inline_3d;
    int32_t crossline_3d;
    
    bool operator==(const TraceCoords& other) const {
        return inline_3d == other.inline_3d && 
               crossline_3d == other.crossline_3d;
    }
};

// Hash function for TraceCoords
struct TraceCoordsHash {
    std::size_t operator()(const TraceCoords& coords) const {
        return std::hash<int32_t>()(coords.inline_3d) ^ 
               (std::hash<int32_t>()(coords.crossline_3d) << 1);
    }
};

// Lookup table: (inline, crossline) -> trace_index
using LookupTable = std::unordered_map<TraceCoords, size_t, TraceCoordsHash>;

// Result of creating lookup table
struct LookupTableResult {
    LookupTable lookup_table;
    std::vector<int32_t> inline_labels;      // Sorted
    std::vector<int32_t> crossline_labels;    // Sorted
    size_t num_traces;
    size_t num_samples;
    uint16_t dt_microseconds;
    double dt_seconds;
};

// Class for working with SEG-Y files
class SegyFile {
public:
    SegyFile(const std::string& filepath);
    ~SegyFile();
    
    // Create lookup table
    LookupTableResult createLookupTable();
    
    // Read trace by coordinates
    std::vector<float> readTraceByCoords(const LookupTable& lookup_table,
                                        int32_t inline_3d, 
                                        int32_t crossline_3d,
                                        size_t num_samples);
    
    // Read inline by coordinates
    std::vector<std::vector<float>> readIlineByCoords(
        const LookupTable& lookup_table,
        int32_t inline_3d,
        const std::vector<int32_t>& crossline_labels,
        size_t num_samples);
    
    // Get metadata
    size_t getNumTraces() const { return num_traces_; }
    size_t getNumSamples() const { return num_samples_; }
    uint16_t getDtMicroseconds() const { return dt_microseconds_; }
    double getDtSeconds() const { return dt_seconds_; }
    uint16_t getDataFormat() const { return data_format_; }
    
    // Read trace header
    void readTraceHeader(size_t trace_idx, std::vector<char>& header);
    
    // Read trace data
    void readTraceData(size_t trace_idx, std::vector<float>& data);
    
    // Get headers for copying
    const std::vector<char>& getTextHeader() const { return text_header_; }
    const std::vector<char>& getBinaryHeader() const { return binary_header_; }
    
    // Create output file
    static void createOutputFile(const std::string& output_path,
                                const LookupTableResult& input_meta,
                                const std::vector<int32_t>& output_inline_labels,
                                const std::vector<int32_t>& output_crossline_labels,
                                const std::vector<char>& text_header,
                                const std::vector<char>& binary_header,
                                uint16_t data_format);
    
    // Write trace to output file
    static void writeTraceToFile(const std::string& output_path,
                                size_t il_idx, size_t xl_idx,
                                const std::vector<int32_t>& output_inline_labels,
                                const std::vector<int32_t>& output_crossline_labels,
                                const std::vector<float>& trace_data,
                                const LookupTableResult& input_meta,
                                const std::vector<char>& input_trace_header,
                                uint16_t data_format);
    
private:
    std::string filepath_;
    std::ifstream file_;
    size_t num_traces_;
    size_t num_samples_;
    uint16_t dt_microseconds_;
    double dt_seconds_;
    uint16_t data_format_;                // 1 = IBM Float, 5 = IEEE Float
    std::vector<char> text_header_;      // 3200 bytes
    std::vector<char> binary_header_;     // 400 bytes
    
    static const size_t TEXT_HEADER_SIZE = 3200;
    static const size_t BINARY_HEADER_SIZE = 400;
    static const size_t TRACE_HEADER_SIZE = 240;
    static const size_t SEGY_HEADER_SIZE = 3600; // 3200 + 400
    
    // Access to constants for SegyOutputFile
    friend class SegyOutputFile;
    
    void readFileMetadata();
    int32_t readInt32FromHeader(const std::vector<char>& header, size_t offset);
    uint16_t swapBytes16(uint16_t val) const;
    uint32_t swapBytes32(uint32_t val) const;
    float swapBytesFloat(float val) const;
    float ibmToIeee(uint32_t ibm);
    static uint32_t ieeeToIbm(float val);
    size_t getTraceOffset(size_t trace_idx) const;
};

// Class for efficient writing of output SEG-Y file
// Keeps file open for fast writing
class SegyOutputFile {
public:
    SegyOutputFile(const std::string& output_path,
                   const LookupTableResult& input_meta,
                   const std::vector<int32_t>& output_inline_labels,
                   const std::vector<int32_t>& output_crossline_labels,
                   const std::vector<char>& text_header,
                   const std::vector<char>& binary_header,
                   uint16_t data_format);
    
    ~SegyOutputFile();
    
    // Write trace (fast version with open file)
    void writeTrace(size_t il_idx, size_t xl_idx,
                   const std::vector<float>& trace_data,
                   const std::vector<char>& input_trace_header);
    
    // Close file (called automatically in destructor)
    void close();
    
private:
    std::string output_path_;
    std::fstream output_file_;
    size_t n_xl_out_;
    size_t trace_data_size_;
    size_t full_trace_size_;
    uint16_t data_format_;
    size_t num_samples_;
    std::vector<int32_t> output_inline_labels_;
    std::vector<int32_t> output_crossline_labels_;
    
    static uint32_t ieeeToIbm(float val);
    static uint32_t swapBytes32(uint32_t val);
};

#endif // SEGY_UTILS_H

