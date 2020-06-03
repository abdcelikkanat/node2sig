project(${CMAKE_PROJECT_NAME}_tests)

# Clone Google Test
set(GOOGLETEST_DIR ${CMAKE_SOURCE_DIR}/tests/lib/googletest)
if(NOT EXISTS ${GOOGLETEST_DIR})
    find_package(Git REQUIRED)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} clone https://github.com/google/googletest ${GOOGLETEST_DIR}
    )
endif()



add_subdirectory(lib/googletest)


set(SOURCE_FILES ../src/Graph.cpp)
set(LIB_FILES ../lib/Graph.h)

add_executable(tests test_matrix.cpp ${SOURCE_FILES} ${LIB_FILES})
target_link_libraries(tests gtest gtest_main)
