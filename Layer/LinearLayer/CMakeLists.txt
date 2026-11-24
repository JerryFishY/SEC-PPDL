project(LinearLayer_test)
find_package(Threads REQUIRED)

find_package(HEXL 1.2.6 REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../Extern/hexl/hexl_build" NO_DEFAULT_PATH)
message(STATUS "HEXL_FOUND: ${HEXL_FOUND}")


file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(LinearLayer STATIC ${srcs})
target_include_directories(LinearLayer PUBLIC include)

target_link_libraries(LinearLayer PUBLIC
    Threads::Threads
    LinearOperator NonlinearOperator
    Utils
    HEXL::hexl
)

message(STATUS "LinearLayer_FOUND: ${LinearLayer_FOUND}")
if(USE_HE_GPU)
    find_package(CUDAToolkit REQUIRED)
    add_library(OptiEncodeRingGPU STATIC src/RingConv.cu)
    set_target_properties(OptiEncodeRingGPU PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(OptiEncodeRingGPU PRIVATE CUDA::cudart)
    target_link_libraries(LinearLayer PRIVATE OptiEncodeRingGPU)
endif()
