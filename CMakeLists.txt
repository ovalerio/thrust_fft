project(thrust_fft)
cmake_minimum_required(VERSION 3.2)

find_package(CUDA REQUIRED)

set(CUDA_VERBOSE_BUILD ON)

find_library(CUFFT_STATIC_LIBRARY
    names cufft_static
    HINTS "/usr/local/cuda-6.5/lib64" "/usr/local/cuda-6.5/lib"
)

find_library(CULIBOS_LIBRARY
    name culibos
    HINTS "/usr/local/cuda-6.5/lib64" "/usr/local/cuda-6.5/lib"
)

# set source files
set(PROJECT_SRC 
	src/thrust_fft_example.cu)

# build options
set(NVCC_OPTS 
	-gencode=arch=compute_35,code=sm_35
)

#link_directories( /usr/local/cuda-6.5/lib64 )

#NOTE: if CMAKE_CXX_FLAGS is missing cuda_link_separable_compilation_objects fails
set(CMAKE_CXX_FLAGS "-g -L/usr/local/cuda-6.5/lib64")
set(CUDA_SEPARABLE_COMPILATION ON)
cuda_compile(DEVICE_OBJS ${PROJECT_SRC} OPTIONS ${NVCC_OPTS})
cuda_compute_separable_compilation_object_file_name(LINK_OBJS thrust_fft ${DEVICE_OBJS})
cuda_link_separable_compilation_objects(${LINK_OBJS} thrust_fft ${NVCC_OPTS} ${DEVICE_OBJS} 
	${CUDA_LIBRARIES}
	${CUFFT_STATIC_LIBRARY}
	${CULIBOS_LIBRARY}
)

add_executable(thrust_fft 
               ${DEVICE_OBJS}
               ${LINK_OBJS})
            
target_link_libraries(thrust_fft 
	${CUDA_LIBRARIES}
	${CUFFT_STATIC_LIBRARY}
	${CULIBOS_LIBRARY}
)

set_target_properties(thrust_fft 
                      PROPERTIES
                      LINKER_LANGUAGE ${CUDA_C_OR_CXX})

install(TARGETS thrust_fft
        RUNTIME DESTINATION ${CMAKE_BINARY_DIR})
