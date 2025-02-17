cmake_minimum_required(VERSION 3.5.0)
project(GPU_Primer VERSION 0.1.0 LANGUAGES C CXX CUDA)

cmake_policy(SET CMP0146 NEW)
find_package(CUDAToolkit REQUIRED)

add_executable(primer primer.cu)


# Set CUDA standard
set_property(TARGET primer PROPERTY CUDA_STANDARD 17)
set_property(TARGET primer PROPERTY CUDA_STANDARD_REQUIRED TRUE)

# Common compile flags
set(CUDA_COMMON_FLAGS
    -arch=sm_89 # Replace with your target architecture
)

# Debug configuration
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Building in Debug mode")
    target_compile_options(primer PRIVATE
        ${CUDA_COMMON_FLAGS}
        -g # Enable debug information
    )
endif()

# Release configuration
if(CMAKE_BUILD_TYPE MATCHES "Release")
    message(STATUS "Building in Release mode")
    target_compile_options(primer PRIVATE
        ${CUDA_COMMON_FLAGS}
        -O3 # Enable optimization
    )
endif()

target_link_libraries(primer ${CUDA_LIBRARIES})

