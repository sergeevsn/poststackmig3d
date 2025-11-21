cmake_minimum_required(VERSION 3.10)
project(poststackmig CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(ENABLE_SIMD "Enable SIMD optimizations" ON)

# Find OpenMP
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found, parallelization will be disabled")
        set(ENABLE_OPENMP OFF)
    endif()
endif()

# Compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native -ffast-math -funroll-loops")

# SIMD flags
if(ENABLE_SIMD)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        message(STATUS "SIMD: AVX2 enabled")
        add_definitions(-DUSE_SIMD)
    else()
        message(STATUS "SIMD: Not supported on this architecture")
        set(ENABLE_SIMD OFF)
    endif()
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/config_parser.cpp
    src/segy_utils.cpp
    src/velocity_reader.cpp
    src/migration_kernel.cpp
    src/progress_bar.cpp
)

set(HEADERS
    include/config_parser.h
    include/segy_utils.h
    include/velocity_reader.h
    include/migration_kernel.h
    include/progress_bar.h
)

# Executable
add_executable(poststackmig
    ${SOURCES}
    ${HEADERS}
)

# Link OpenMP
if(ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(poststackmig PUBLIC OpenMP::OpenMP_CXX)
    add_definitions(-DUSE_OPENMP)
endif()

# Include directories
target_include_directories(poststackmig PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Installation
install(TARGETS poststackmig DESTINATION bin)

