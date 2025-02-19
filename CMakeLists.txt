cmake_minimum_required(VERSION 3.5.0)
project(GPU_Primer VERSION 0.1.0 LANGUAGES C CXX CUDA)

cmake_policy(SET CMP0146 NEW)
find_package(CUDAToolkit REQUIRED)

add_executable(mnist mnist.cu kernels.cu utils.cu dataset.cpp)

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
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS} -g> # CUDA-specific flags
            $<$<COMPILE_LANGUAGE:CXX>:-g> # C++-specific flags
        )
    elseif(CMAKE_BUILD_TYPE MATCHES "Release")
        message(STATUS "Building ${target} in Release mode")
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS} -O3> # CUDA-specific flags
            $<$<COMPILE_LANGUAGE:CXX>:-O3> # C++-specific flags
        )
    endif()
endfunction()

# Apply the function to each target
setup_target(mnist)

