cmake_minimum_required(VERSION 3.12.3)
project(nodesig)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)


set(MKLROOT /opt/intel/mkl/)
set(ENV{MKLROOT} /opt/intel/mkl/)
set(MKL_FIND_DEBUG ON)
set(MKL_USE_STATIC_LIBS ON)
find_package(MKL REQUIRED)
if(MKL_FOUND)
    message("MKL Found!")
else()
    message(WARNING "MKL libs not found")
endif()



#cmake_minimum_required(VERSION 3.9)
project(nodesig LANGUAGES CXX)



include_directories(lib)
include_directories(lib/eigen)
include_directories(${MKL_INCLUDE_DIRS})

add_subdirectory(src)

set(SOURCE_FILES src/Graph.cpp)
set(LIB_FILES lib/Graph.h lib/Model.h lib/Utilities.h)

add_executable(${CMAKE_PROJECT_NAME} main.cpp ${SOURCE_FILES} ${LIB_FILES})

################ OpenMP ################
find_package(OpenMP REQUIRED)
########################################
#target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl)
# Clone Eigen Lib
set(EIGEN_DIR ${CMAKE_SOURCE_DIR}/lib/eigen)
if(NOT EXISTS ${EIGEN_DIR})
    find_package(Git REQUIRED)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} clone https://gitlab.com/libeigen/eigen.git ${EIGEN_DIR}
    )
endif()



