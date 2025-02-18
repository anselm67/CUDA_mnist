cmake_minimum_required(VERSION 3.5.0)
project(GPU_Primer VERSION 0.1.0 LANGUAGES C CXX CUDA)

cmake_policy(SET CMP0146 NEW)
find_package(CUDAToolkit REQUIRED)

add_executable(primer primer.cu)
add_executable(mult mult.cu)
add_executable(mnist mnist.cu)

# Common compile flags
set(CUDA_COMMON_FLAGS
    -arch=sm_89 # Replace with your target architecture
)

# Function to set common properties and compile options
function(setup_target target)
    set_property(TARGET ${target} PROPERTY CUDA_STANDARD 17)
    set_property(TARGET ${target} PROPERTY CUDA_STANDARD_REQUIRED TRUE)
    target_link_libraries(${target} ${CUDA_LIBRARIES})

    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        message(STATUS "Building ${target} in Debug mode")
        target_compile_options(${target} PRIVATE
            ${CUDA_COMMON_FLAGS}
            -g # Enable debug information
        )
    elseif(CMAKE_BUILD_TYPE MATCHES "Release")
        message(STATUS "Building ${target} in Release mode")
        target_compile_options(${target} PRIVATE
            ${CUDA_COMMON_FLAGS}
            -O3 # Enable optimization
        )
    endif()
endfunction()

# Apply the function to each target
setup_target(primer)
setup_target(mult)
setup_target(mnist)

