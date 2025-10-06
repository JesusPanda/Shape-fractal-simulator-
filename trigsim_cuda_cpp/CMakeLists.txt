cmake_minimum_required(VERSION 3.20)
project(TrigsimCUDA CXX CUDA)

# --- Standard and Compiler Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Find Dependencies ---
# CUDA
find_package(CUDA REQUIRED)
# SFML
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# --- Add Executable ---
add_executable(trigsim_cuda main.cpp kernels.cu)

# --- Link Libraries ---
target_link_libraries(trigsim_cuda PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
)

# --- Include Directories ---
target_include_directories(trigsim_cuda PRIVATE
    ${SFML_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- Compiler-Specific Flags (Optional but Recommended) ---
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(trigsim_cuda PRIVATE -O3 -Wall)
elseif(MSVC)
    target_compile_options(trigsim_cuda PRIVATE /O2 /W3)
endif()

# --- CUDA-Specific Settings ---
set_target_properties(trigsim_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)