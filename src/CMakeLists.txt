set(
    SOURCE_FILES
    main.cpp
    integrate.cu
)

find_package(CUDAToolkit REQUIRED)

# FIXME: we kind of just hard-coded the math library with the 
# the n body simulation engine
message(${CUDAToolkit_INCLUDE_DIRS})
add_executable(main ${SOURCE_FILES})
target_include_directories(main PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
