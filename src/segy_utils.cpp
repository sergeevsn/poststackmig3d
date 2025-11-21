#include "segy_utils.h"
#include "progress_bar.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <set>
#include <fstream>
#include <cfloat>
#include <climits>
#include <utility>

#ifdef USE_SIMD
#include <immintrin.h>
#include <tmmintrin.h>  // For _mm_shuffle_epi8
#endif

SegyFile::SegyFile(const std::string& filepath)
    : filepath_(filepath)
    , num_traces_(0)
    , num_samples_(0)
    , dt_microseconds_(0)
    , dt_seconds_(0.0)
    , data_format_(0)
{
    readFileMetadata();
}

SegyFile::~SegyFile() {
    if (file_.is_open()) {
        file_.close();
    }
}

void SegyFile::readFileMetadata() {
    file_.open(filepath_, std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot open SEGY file: " + filepath_);
    }
    
    // Read text header (3200 bytes)
    text_header_.resize(TEXT_HEADER_SIZE);
    file_.read(text_header_.data(), TEXT_HEADER_SIZE);
    if (file_.gcount() != TEXT_HEADER_SIZE) {
        throw std::runtime_error("Failed to read text header");
    }
    
    // Read binary header (400 bytes)
    binary_header_.resize(BINARY_HEADER_SIZE);
    file_.read(binary_header_.data(), BINARY_HEADER_SIZE);
    if (file_.gcount() != BINARY_HEADER_SIZE) {
        throw std::runtime_error("Failed to read binary header");
    }
    
    // Extract sample interval (dt) from binary header (offset 16, 2 bytes)
    std::memcpy(&dt_microseconds_, binary_header_.data() + 16, sizeof(dt_microseconds_));
    dt_microseconds_ = swapBytes16(dt_microseconds_);
    
    if (dt_microseconds_ == 0) {
        throw std::runtime_error("Sample interval (dt) is zero in binary header");
    }
    
    dt_seconds_ = dt_microseconds_ * 1e-6; // Convert microseconds to seconds
    
    // Extract number of samples per trace (offset 20, 2 bytes)
    uint16_t n_samples_per_trace;
    std::memcpy(&n_samples_per_trace, binary_header_.data() + 20, sizeof(n_samples_per_trace));
    n_samples_per_trace = swapBytes16(n_samples_per_trace);
    
    if (n_samples_per_trace == 0) {
        throw std::runtime_error("Number of samples per trace is zero in binary header");
    }
    
    num_samples_ = n_samples_per_trace;
    
    // Extract data format from binary header
    // Offset 3224 from file start = 3200 (text header) + 24 (offset in binary header)
    // Format: 1 = IBM Float, 5 = IEEE Float
    uint16_t format_code;
    std::memcpy(&format_code, binary_header_.data() + 24, sizeof(format_code));
    format_code = swapBytes16(format_code);
    data_format_ = format_code;
    
    if (data_format_ != 1 && data_format_ != 5) {
        std::cerr << "Warning: Unknown data format code: " << data_format_ 
                  << ", assuming IEEE Float (5)" << std::endl;
        data_format_ = 5; // Default to IEEE
    }
    
    // Calculate number of traces
    const size_t trace_data_size = num_samples_ * sizeof(uint32_t);
    const size_t full_trace_size = TRACE_HEADER_SIZE + trace_data_size;
    
    file_.seekg(0, std::ios::end);
    std::streampos file_size = file_.tellg();
    file_.seekg(SEGY_HEADER_SIZE); // Start of traces
    
    std::streamoff data_size = file_size - std::streamoff(SEGY_HEADER_SIZE);
    num_traces_ = static_cast<size_t>(data_size / full_trace_size);
    
    if (num_traces_ == 0) {
        throw std::runtime_error("No traces found in SEGY file");
    }
    
    // Reset file pointer to beginning of traces
    file_.seekg(SEGY_HEADER_SIZE);
}

size_t SegyFile::getTraceOffset(size_t trace_idx) const {
    const size_t trace_data_size = num_samples_ * sizeof(uint32_t);
    const size_t full_trace_size = TRACE_HEADER_SIZE + trace_data_size;
    return SEGY_HEADER_SIZE + trace_idx * full_trace_size;
}

uint16_t SegyFile::swapBytes16(uint16_t val) const {
    return (val << 8) | (val >> 8);
}

uint32_t SegyFile::swapBytes32(uint32_t val) const {
    return ((val & 0xFF000000) >> 24) |
           ((val & 0x00FF0000) >> 8) |
           ((val & 0x0000FF00) << 8) |
           ((val & 0x000000FF) << 24);
}

float SegyFile::swapBytesFloat(float val) const {
    uint32_t int_val;
    std::memcpy(&int_val, &val, sizeof(float));
    int_val = swapBytes32(int_val);
    float result;
    std::memcpy(&result, &int_val, sizeof(float));
    return result;
}

int32_t SegyFile::readInt32FromHeader(const std::vector<char>& header, size_t offset) {
    if (offset + sizeof(int32_t) > header.size()) {
        throw std::runtime_error("Header offset out of bounds");
    }
    
    int32_t value;
    std::memcpy(&value, header.data() + offset, sizeof(int32_t));
    // SEG-Y uses big-endian, so swap if needed
    // For simplicity, assume we need to swap (can be made configurable)
    value = static_cast<int32_t>(swapBytes32(static_cast<uint32_t>(value)));
    return value;
}

LookupTableResult SegyFile::createLookupTable() {
    LookupTableResult result;
    result.num_traces = num_traces_;
    result.num_samples = num_samples_;
    result.dt_microseconds = dt_microseconds_;
    result.dt_seconds = dt_seconds_;
    
    std::set<int32_t> inline_set;
    std::set<int32_t> crossline_set;
    
    ProgressBar progress("Creating tracemap", num_traces_);
    
    for (size_t trace_idx = 0; trace_idx < num_traces_; ++trace_idx) {
        // Read trace header
        std::vector<char> trace_header(TRACE_HEADER_SIZE);
        size_t trace_offset = getTraceOffset(trace_idx);
        
        file_.seekg(trace_offset);
        file_.read(trace_header.data(), TRACE_HEADER_SIZE);
        
        if (file_.gcount() != TRACE_HEADER_SIZE) {
            throw std::runtime_error("Failed to read trace header " + 
                                   std::to_string(trace_idx));
        }
        
        // Extract INLINE_3D (offset 188, 4 bytes) and CROSSLINE_3D (offset 192, 4 bytes)
        // Note: SEG-Y trace header offsets are 0-based in the 240-byte header
        int32_t inline_3d = readInt32FromHeader(trace_header, 188);
        int32_t crossline_3d = readInt32FromHeader(trace_header, 192);
        
        if (inline_3d == 0 || crossline_3d == 0) {
            // Skip traces with zero coordinates
            continue;
        }
        
        TraceCoords coords;
        coords.inline_3d = inline_3d;
        coords.crossline_3d = crossline_3d;
        
        result.lookup_table[coords] = trace_idx;
        inline_set.insert(inline_3d);
        crossline_set.insert(crossline_3d);
        
        progress.update(trace_idx + 1);
    }
    
    progress.finish();
    
    // Convert sets to sorted vectors
    result.inline_labels.assign(inline_set.begin(), inline_set.end());
    result.crossline_labels.assign(crossline_set.begin(), crossline_set.end());
    std::sort(result.inline_labels.begin(), result.inline_labels.end());
    std::sort(result.crossline_labels.begin(), result.crossline_labels.end());
    
    std::cout << "Lookup table created: " << result.lookup_table.size() 
              << " traces, " << result.inline_labels.size() << " inlines, "
              << result.crossline_labels.size() << " crosslines" << std::endl;
    std::cout << "Data format: " << data_format_ 
              << (data_format_ == 1 ? " (IBM Float)" : 
                  data_format_ == 5 ? " (IEEE Float)" : " (Unknown)") << std::endl;
    
    return result;
}

std::vector<float> SegyFile::readTraceByCoords(const LookupTable& lookup_table,
                                              int32_t inline_3d, 
                                              int32_t crossline_3d,
                                              size_t num_samples) {
    TraceCoords coords;
    coords.inline_3d = inline_3d;
    coords.crossline_3d = crossline_3d;
    
    auto it = lookup_table.find(coords);
    if (it == lookup_table.end()) {
        // Return zero trace if not found
        return std::vector<float>(num_samples, 0.0f);
    }
    
    size_t trace_idx = it->second;
    std::vector<float> trace_data;
    readTraceData(trace_idx, trace_data);
    
    // Ensure correct size
    if (trace_data.size() != num_samples) {
        trace_data.resize(num_samples, 0.0f);
    }
    
    return trace_data;
}

// Convert IBM float (32-bit) to IEEE float
float SegyFile::ibmToIeee(uint32_t ibm) {
    if (ibm == 0) return 0.0f;
    
    int sign = ((ibm >> 31) & 0x01);
    int exponent = ((ibm >> 24) & 0x7F) - 64;
    uint32_t fraction = ibm & 0x00FFFFFF;
    
    double mantissa = static_cast<double>(fraction) / static_cast<double>(0x01000000);
    double value = std::ldexp(mantissa, exponent * 4);
    
    return sign ? -static_cast<float>(value) : static_cast<float>(value);
}

// Convert IEEE float to IBM float (for writing)
uint32_t SegyFile::ieeeToIbm(float val) {
    if (val == 0.0f) return 0;
    
    uint32_t sign = (val < 0) ? 0x80000000 : 0;
    if (val < 0) val = -val;
    
    int exponent = 64;
    while (val < 1.0f) { val *= 16.0f; --exponent; }
    while (val >= 1.0f) { val /= 16.0f; ++exponent; }
    
    uint32_t fraction = static_cast<uint32_t>(val * 0x01000000) & 0x00FFFFFF;
    
    return sign | (exponent << 24) | fraction;
}

void SegyFile::readTraceData(size_t trace_idx, std::vector<float>& data) {
    size_t trace_offset = getTraceOffset(trace_idx);
    size_t data_offset = trace_offset + TRACE_HEADER_SIZE;
    
    file_.seekg(data_offset);
    
    data.resize(num_samples_);
    
    const size_t trace_data_size = num_samples_ * sizeof(uint32_t);
    std::vector<uint32_t> raw_buffer(num_samples_);
    
    // Optimization: read all data at once
    file_.read(reinterpret_cast<char*>(raw_buffer.data()), trace_data_size);
    
    if (file_.gcount() != static_cast<std::streamsize>(trace_data_size)) {
        throw std::runtime_error("Failed to read trace data " + std::to_string(trace_idx));
    }
    
    // Process data
    if (data_format_ == 1) {
        // IBM Float format
        for (size_t i = 0; i < num_samples_; ++i) {
            uint32_t sample_swapped = swapBytes32(raw_buffer[i]);
            data[i] = ibmToIeee(sample_swapped);
        }
    } else if (data_format_ == 5) {
        // IEEE Float format - optimized reading
        for (size_t i = 0; i < num_samples_; ++i) {
            uint32_t sample_swapped = swapBytes32(raw_buffer[i]);
            float sample_ieee;
            std::memcpy(&sample_ieee, &sample_swapped, sizeof(float));
            data[i] = sample_ieee;
        }
    } else {
        // Unknown format, try IEEE first, then IBM
        for (size_t i = 0; i < num_samples_; ++i) {
            uint32_t sample_swapped = swapBytes32(raw_buffer[i]);
            float sample_ieee;
            std::memcpy(&sample_ieee, &sample_swapped, sizeof(float));
            if (std::isnan(sample_ieee) || std::isinf(sample_ieee)) {
                data[i] = ibmToIeee(sample_swapped);
            } else {
                data[i] = sample_ieee;
            }
        }
    }
}

std::vector<std::vector<float>> SegyFile::readIlineByCoords(
    const LookupTable& lookup_table,
    int32_t inline_3d,
    const std::vector<int32_t>& crossline_labels,
    size_t num_samples) {
    
    std::vector<std::vector<float>> iline_data;
    iline_data.reserve(crossline_labels.size());
    
    // Optimization: collect all trace indices for this inline
    // and read them sequentially for better performance
    std::vector<size_t> trace_indices;
    trace_indices.reserve(crossline_labels.size());
    
    for (int32_t crossline_3d : crossline_labels) {
        TraceCoords coords;
        coords.inline_3d = inline_3d;
        coords.crossline_3d = crossline_3d;
        
        auto it = lookup_table.find(coords);
        if (it != lookup_table.end()) {
            trace_indices.push_back(it->second);
        } else {
            trace_indices.push_back(SIZE_MAX); // Marker for missing trace
        }
    }
    
    // Sort indices for sequential reading (if possible)
    // But preserve original order for result
    std::vector<std::pair<size_t, size_t>> indexed_order;
    indexed_order.reserve(trace_indices.size());
    for (size_t i = 0; i < trace_indices.size(); ++i) {
        if (trace_indices[i] != SIZE_MAX) {
            indexed_order.push_back({trace_indices[i], i});
        }
    }
    
    // Sort by trace index for sequential reading
    std::sort(indexed_order.begin(), indexed_order.end(),
              [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
                  return a.first < b.first;
              });
    
    // Initialize result
    iline_data.resize(crossline_labels.size());
    for (size_t i = 0; i < crossline_labels.size(); ++i) {
        iline_data[i].resize(num_samples, 0.0f);
    }
    
    // Read traces sequentially (in file order)
    for (const auto& pair : indexed_order) {
        size_t trace_idx = pair.first;
        size_t result_idx = pair.second;
        
        std::vector<float> trace_data;
        readTraceData(trace_idx, trace_data);
        
        if (trace_data.size() == num_samples) {
            iline_data[result_idx] = std::move(trace_data);
        }
    }
    
    return iline_data;
}

void SegyFile::readTraceHeader(size_t trace_idx, std::vector<char>& header) {
    header.resize(TRACE_HEADER_SIZE);
    size_t trace_offset = getTraceOffset(trace_idx);
    
    file_.seekg(trace_offset);
    file_.read(header.data(), TRACE_HEADER_SIZE);
    
    if (file_.gcount() != TRACE_HEADER_SIZE) {
        throw std::runtime_error("Failed to read trace header " + 
                               std::to_string(trace_idx));
    }
}

void SegyFile::createOutputFile(const std::string& output_path,
                               const LookupTableResult& input_meta,
                               const std::vector<int32_t>& output_inline_labels,
                               const std::vector<int32_t>& output_crossline_labels,
                               const std::vector<char>& text_header,
                               const std::vector<char>& binary_header,
                               uint16_t data_format) {
    std::ofstream output_file(output_path, std::ios::binary);
    if (!output_file.is_open()) {
        throw std::runtime_error("Cannot create output SEGY file: " + output_path);
    }
    
    // Write text header
    output_file.write(text_header.data(), text_header.size());
    
    // Write binary header (update number of samples and data format)
    std::vector<char> binary_header_copy = binary_header;
    uint16_t n_samples = static_cast<uint16_t>(input_meta.num_samples);
    
    // Swap bytes for big-endian
    uint16_t n_samples_swapped = (n_samples << 8) | (n_samples >> 8);
    std::memcpy(binary_header_copy.data() + 20, &n_samples_swapped, sizeof(n_samples_swapped));
    
    // Update data format (offset 24 in binary header = offset 3224 from file start)
    uint16_t format_swapped = (data_format << 8) | (data_format >> 8);
    std::memcpy(binary_header_copy.data() + 24, &format_swapped, sizeof(format_swapped));
    
    output_file.write(binary_header_copy.data(), binary_header_copy.size());
    
    // Calculate output dimensions
    size_t n_il_out = output_inline_labels.size();
    size_t n_xl_out = output_crossline_labels.size();
    
    // Initialize trace headers and data
    std::vector<char> empty_header(TRACE_HEADER_SIZE, 0);
    std::vector<uint32_t> empty_data(input_meta.num_samples, 0);
    
    for (size_t il_idx = 0; il_idx < n_il_out; ++il_idx) {
        for (size_t xl_idx = 0; xl_idx < n_xl_out; ++xl_idx) {
            // Write trace header
            std::vector<char> header = empty_header;
            
            // Set INLINE_3D and CROSSLINE_3D
            int32_t inline_3d = output_inline_labels[il_idx];
            int32_t crossline_3d = output_crossline_labels[xl_idx];
            
            // Swap bytes for big-endian
            uint32_t inline_swapped = ((static_cast<uint32_t>(inline_3d) & 0xFF000000) >> 24) |
                                      ((static_cast<uint32_t>(inline_3d) & 0x00FF0000) >> 8) |
                                      ((static_cast<uint32_t>(inline_3d) & 0x0000FF00) << 8) |
                                      ((static_cast<uint32_t>(inline_3d) & 0x000000FF) << 24);
            
            uint32_t crossline_swapped = ((static_cast<uint32_t>(crossline_3d) & 0xFF000000) >> 24) |
                                         ((static_cast<uint32_t>(crossline_3d) & 0x00FF0000) >> 8) |
                                         ((static_cast<uint32_t>(crossline_3d) & 0x0000FF00) << 8) |
                                         ((static_cast<uint32_t>(crossline_3d) & 0x000000FF) << 24);
            
            std::memcpy(header.data() + 188, &inline_swapped, sizeof(int32_t));
            std::memcpy(header.data() + 192, &crossline_swapped, sizeof(int32_t));
            
            output_file.write(header.data(), TRACE_HEADER_SIZE);
            
            // Write zero trace data
            for (size_t i = 0; i < input_meta.num_samples; ++i) {
                uint32_t zero_swapped = 0;
                output_file.write(reinterpret_cast<const char*>(&zero_swapped), 
                                 sizeof(uint32_t));
            }
        }
    }
    
    output_file.close();
    
    std::cout << "Output file created: " << output_path << std::endl;
    std::cout << "Output dimensions: IL=" << n_il_out 
              << ", XL=" << n_xl_out << std::endl;
}

void SegyFile::writeTraceToFile(const std::string& output_path,
                               size_t il_idx, size_t xl_idx,
                               const std::vector<int32_t>& output_inline_labels,
                               const std::vector<int32_t>& output_crossline_labels,
                               const std::vector<float>& trace_data,
                               const LookupTableResult& input_meta,
                               const std::vector<char>& input_trace_header,
                               uint16_t data_format) {
    std::fstream output_file(output_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!output_file.is_open()) {
        throw std::runtime_error("Cannot open output SEGY file for writing: " + output_path);
    }
    
    size_t n_xl_out = output_crossline_labels.size();
    size_t trace_idx = il_idx * n_xl_out + xl_idx;
    
    const size_t trace_data_size = input_meta.num_samples * sizeof(uint32_t);
    const size_t full_trace_size = TRACE_HEADER_SIZE + trace_data_size;
    size_t trace_offset = SEGY_HEADER_SIZE + trace_idx * full_trace_size;
    
    // Write trace header
    output_file.seekp(trace_offset);
    if (input_trace_header.size() == TRACE_HEADER_SIZE) {
        output_file.write(input_trace_header.data(), TRACE_HEADER_SIZE);
    } else {
        // Create default header
        std::vector<char> header(TRACE_HEADER_SIZE, 0);
        int32_t inline_3d = output_inline_labels[il_idx];
        int32_t crossline_3d = output_crossline_labels[xl_idx];
        
        uint32_t inline_swapped = ((static_cast<uint32_t>(inline_3d) & 0xFF000000) >> 24) |
                                  ((static_cast<uint32_t>(inline_3d) & 0x00FF0000) >> 8) |
                                  ((static_cast<uint32_t>(inline_3d) & 0x0000FF00) << 8) |
                                  ((static_cast<uint32_t>(inline_3d) & 0x000000FF) << 24);
        
        uint32_t crossline_swapped = ((static_cast<uint32_t>(crossline_3d) & 0xFF000000) >> 24) |
                                     ((static_cast<uint32_t>(crossline_3d) & 0x00FF0000) >> 8) |
                                     ((static_cast<uint32_t>(crossline_3d) & 0x0000FF00) << 8) |
                                     ((static_cast<uint32_t>(crossline_3d) & 0x000000FF) << 24);
        
        std::memcpy(header.data() + 188, &inline_swapped, sizeof(int32_t));
        std::memcpy(header.data() + 192, &crossline_swapped, sizeof(int32_t));
        output_file.write(header.data(), TRACE_HEADER_SIZE);
    }
    
    // Write trace data
    size_t data_offset = trace_offset + TRACE_HEADER_SIZE;
    output_file.seekp(data_offset);
    
    for (size_t i = 0; i < trace_data.size() && i < input_meta.num_samples; ++i) {
        float sample = trace_data[i];
        uint32_t sample_raw;
        
        // Convert to appropriate format based on data_format
        if (data_format == 1) {
            // IBM Float format
            sample_raw = ieeeToIbm(sample);
        } else {
            // IEEE Float format (or default)
            std::memcpy(&sample_raw, &sample, sizeof(float));
        }
        
        // Swap bytes for big-endian
        sample_raw = ((sample_raw & 0xFF000000) >> 24) |
                     ((sample_raw & 0x00FF0000) >> 8) |
                     ((sample_raw & 0x0000FF00) << 8) |
                     ((sample_raw & 0x000000FF) << 24);
        
        output_file.write(reinterpret_cast<const char*>(&sample_raw), sizeof(uint32_t));
    }
    
    output_file.close();
}

// SegyOutputFile implementation
SegyOutputFile::SegyOutputFile(const std::string& output_path,
                               const LookupTableResult& input_meta,
                               const std::vector<int32_t>& output_inline_labels,
                               const std::vector<int32_t>& output_crossline_labels,
                               const std::vector<char>& /* text_header */,
                               const std::vector<char>& /* binary_header */,
                               uint16_t data_format)
    : output_path_(output_path)
    , n_xl_out_(output_crossline_labels.size())
    , trace_data_size_(input_meta.num_samples * sizeof(uint32_t))
    , full_trace_size_(SegyFile::TRACE_HEADER_SIZE + trace_data_size_)
    , data_format_(data_format)
    , num_samples_(input_meta.num_samples)
    , output_inline_labels_(output_inline_labels)
    , output_crossline_labels_(output_crossline_labels)
{
    output_file_.open(output_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!output_file_.is_open()) {
        throw std::runtime_error("Cannot open output SEGY file for writing: " + output_path);
    }
    
    // File already created in createOutputFile, just open for writing
}

SegyOutputFile::~SegyOutputFile() {
    close();
}

void SegyOutputFile::close() {
    if (output_file_.is_open()) {
        output_file_.close();
    }
}

void SegyOutputFile::writeTrace(size_t il_idx, size_t xl_idx,
                                const std::vector<float>& trace_data,
                                const std::vector<char>& input_trace_header) {
    size_t trace_idx = il_idx * n_xl_out_ + xl_idx;
    size_t trace_offset = SegyFile::SEGY_HEADER_SIZE + trace_idx * full_trace_size_;
    
    // Write trace header
    output_file_.seekp(trace_offset);
    if (input_trace_header.size() == SegyFile::TRACE_HEADER_SIZE) {
        output_file_.write(input_trace_header.data(), SegyFile::TRACE_HEADER_SIZE);
    } else {
        // Create default header
        std::vector<char> header(SegyFile::TRACE_HEADER_SIZE, 0);
        int32_t inline_3d = output_inline_labels_[il_idx];
        int32_t crossline_3d = output_crossline_labels_[xl_idx];
        
        uint32_t inline_swapped = swapBytes32(static_cast<uint32_t>(inline_3d));
        uint32_t crossline_swapped = swapBytes32(static_cast<uint32_t>(crossline_3d));
        
        std::memcpy(header.data() + 188, &inline_swapped, sizeof(int32_t));
        std::memcpy(header.data() + 192, &crossline_swapped, sizeof(int32_t));
        output_file_.write(header.data(), SegyFile::TRACE_HEADER_SIZE);
    }
    
    // Write trace data
    size_t data_offset = trace_offset + SegyFile::TRACE_HEADER_SIZE;
    output_file_.seekp(data_offset);
    
    // Optimization: write data in batch
    std::vector<uint32_t> raw_buffer(num_samples_);
    
    for (size_t i = 0; i < trace_data.size() && i < num_samples_; ++i) {
        float sample = trace_data[i];
        uint32_t sample_raw;
        
        // Convert to appropriate format based on data_format
        if (data_format_ == 1) {
            // IBM Float format
            sample_raw = ieeeToIbm(sample);
        } else {
            // IEEE Float format (or default)
            std::memcpy(&sample_raw, &sample, sizeof(float));
        }
        
        // Swap bytes for big-endian
        raw_buffer[i] = swapBytes32(sample_raw);
    }
    
    // Write all data at once
    output_file_.write(reinterpret_cast<const char*>(raw_buffer.data()), trace_data_size_);
}

uint32_t SegyOutputFile::swapBytes32(uint32_t val) {
    return ((val & 0xFF000000) >> 24) |
           ((val & 0x00FF0000) >> 8) |
           ((val & 0x0000FF00) << 8) |
           ((val & 0x000000FF) << 24);
}

uint32_t SegyOutputFile::ieeeToIbm(float val) {
    return SegyFile::ieeeToIbm(val);
}

