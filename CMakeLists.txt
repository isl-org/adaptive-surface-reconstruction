cmake_minimum_required( VERSION 3.18 )

project(adaptivesurfacereconstruction VERSION 0.1.0)

if( NOT CMAKE_BUILD_TYPE )
        set( CMAKE_BUILD_TYPE "Release" CACHE STRING "Build configuration 'Release' or 'Debug'." FORCE )
endif()

# enable all warnings
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall" )

include(cmake/external_deps.cmake)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_subdirectory( cpp/lib lib ) 
add_subdirectory( cpp/bin bin ) 
add_subdirectory( cpp/pybind pybind ) 

add_subdirectory( python ) 
add_subdirectory( appimage ) 

add_custom_target( apply-style 
        COMMAND ${CMAKE_COMMAND}
        -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        -DDO_APPLY_STYLE=ON
        -P ${CMAKE_CURRENT_SOURCE_DIR}/utils/check_cpp_style.cmake
        COMMAND ${Python3_EXECUTABLE} -m yapf --in-place -vv -r --style google "${PROJECT_SOURCE_DIR}/python"
        COMMAND ${Python3_EXECUTABLE} -m yapf --in-place -vv -r --style google "${PROJECT_SOURCE_DIR}/models"
)