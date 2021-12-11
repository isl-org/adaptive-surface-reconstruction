include(FetchContent)
include(ExternalProject)

find_package( Python COMPONENTS Interpreter )

# show downloads during configure step
set( FETCHCONTENT_QUIET OFF CACHE BOOL "" FORCE )

find_program( patchelf_binary patchelf REQUIRED )

if( BUILD_PYTHON_MODULE )
        # download pybind11
        FetchContent_Declare(
                pybind11
                URL "https://github.com/pybind/pybind11/archive/v2.3.0.tar.gz"
        )
        FetchContent_GetProperties(pybind11)
        if(NOT pybind11_POPULATED)
        message( STATUS "Populating pybind11" )
        FetchContent_Populate(pybind11)
        add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
        endif()
endif()


FetchContent_Declare(
        libcuckoo
        GIT_REPOSITORY "https://github.com/efficient/libcuckoo.git"
        GIT_SHALLOW YES
)
FetchContent_GetProperties(libcuckoo)
if(NOT libcuckoo_POPULATED)
  message( STATUS "Populating libcuckoo" )
  FetchContent_Populate(libcuckoo)
  set(libcuckoo_INCLUDE_DIR ${libcuckoo_SOURCE_DIR})
endif()


FetchContent_Declare(
        eigen3
        URL https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
)
FetchContent_GetProperties(eigen3)
if(NOT eigen3_POPULATED)
  message( STATUS "Populating eigen3" )
  FetchContent_Populate(eigen3)
  set(eigen3_INCLUDE_DIR ${eigen3_SOURCE_DIR})
endif()



FetchContent_Declare(
        nanoflann
        URL https://github.com/jlblancoc/nanoflann/archive/v1.3.2.tar.gz
)
FetchContent_GetProperties(nanoflann)
if(NOT nanoflann_POPULATED)
  message( STATUS "Populating nanoflann" )
  FetchContent_Populate(nanoflann)
  set(nanoflann_INCLUDE_DIR ${nanoflann_SOURCE_DIR}/include)
endif()


FetchContent_Declare(
        libtorch
        URL https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-shared-with-deps-1.8.2%2Bcpu.zip
)
FetchContent_GetProperties(libtorch)
if(NOT libtorch_POPULATED)
  message( STATUS "Populating libtorch" )
  FetchContent_Populate(libtorch)
  find_package( Torch REQUIRED PATHS ${libtorch_SOURCE_DIR}/share/cmake/Torch )
  file( GLOB torch_libgomp ${libtorch_SOURCE_DIR}/lib/libgomp*.so.* )
endif()


ExternalProject_Add(
    ext_open3d
    PREFIX open3d
    GIT_REPOSITORY https://github.com/isl-org/Open3D.git
    GIT_TAG v0.14.1
    GIT_SHALLOW YES
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_EXAMPLES=OFF
        -DBUILD_ISPC_MODULE=OFF
        -DBUILD_GUI=OFF
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_PYTHON_MODULE=OFF
        -DBUILD_PYTORCH_OPS=ON
        -DBUILD_WEBRTC=OFF
)
# we patch the runpath of the op library to make the binary work inside the build directory.
ExternalProject_Add_Step(ext_open3d patchelf
        COMMAND ${patchelf_binary} --set-rpath $<TARGET_FILE_DIR:c10> <INSTALL_DIR>/lib/open3d_torch_ops.so
        COMMENT "Patching runpath for open3d_torch_ops"
        DEPENDEES install 
)


ExternalProject_Get_Property(ext_open3d INSTALL_DIR BINARY_DIR)
add_library(Open3DHelper SHARED IMPORTED)
add_dependencies(Open3DHelper ext_open3d)
set_property(TARGET Open3DHelper PROPERTY IMPORTED_LOCATION "${INSTALL_DIR}/lib/libOpen3D.so" )
target_compile_features(Open3DHelper INTERFACE cxx_std_14)
target_compile_definitions(Open3DHelper INTERFACE _GLIBCXX_USE_CXX11_ABI=$<BOOL:${GLIBCXX_USE_CXX11_ABI}>)
set( OPEN3D_INCLUDE_DIRS "${INSTALL_DIR}/include;${INSTALL_DIR}/include/open3d/3rdparty;${BINARY_DIR}/tbb/src/ext_tbb/include" )
# create dirs in configure phase to avoid a non critical cmake error.
file( MAKE_DIRECTORY ${OPEN3D_INCLUDE_DIRS} )
target_include_directories(Open3DHelper INTERFACE ${OPEN3D_INCLUDE_DIRS}
        )
target_link_libraries(Open3DHelper INTERFACE 
        "${BINARY_DIR}/tbb/src/ext_tbb-build/libtbb_static.a"
        "${BINARY_DIR}/tbb/src/ext_tbb-build/libtbbmalloc_static.a"
       )
add_library(Open3D::Open3D ALIAS Open3DHelper)

add_library(open3d_torch_ops SHARED IMPORTED)
set_property(TARGET open3d_torch_ops PROPERTY IMPORTED_LOCATION "${INSTALL_DIR}/lib/open3d_torch_ops.so" )