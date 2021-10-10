file( GLOB asr_SOURCES  *.cpp *.cc *.c )

add_library( asrobjs OBJECT ${asr_SOURCES} )
add_dependencies(asrobjs ext_open3d)

# enable c++14
set_target_properties( asrobjs PROPERTIES 
                CXX_STANDARD 14
                CXX_STANDARD_REQUIRED ON 
                CXX_EXTENSIONS OFF)


# sets GLIBCXX_USE_CXX11_ABI
target_compile_options( asrobjs PUBLIC "${TORCH_CXX_FLAGS}" )
set_target_properties( asrobjs PROPERTIES POSITION_INDEPENDENT_CODE ON )
target_compile_options( asrobjs PRIVATE -Wall -Wextra -Werror )

target_include_directories( asrobjs PUBLIC 
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_include_directories( asrobjs SYSTEM PRIVATE 
        ${libcuckoo_INCLUDE_DIR}
        ${eigen3_INCLUDE_DIR}
        ${nanoflann_INCLUDE_DIR}
        "${TORCH_INCLUDE_DIRS}"
        ${PROJECT_BINARY_DIR}/open3d/include
        ${PROJECT_BINARY_DIR}/open3d/include/open3d/3rdparty
)


configure_file( "asr_config_generated.h.in" "asr_config_generated.h" )
file( READ "${PROJECT_SOURCE_DIR}/3rdparty/README.txt" THIRD_PARTY_NOTICES ) 
configure_file( "third_party_notices.inl.in" "third_party_notices.inl" )


add_library( asr SHARED $<TARGET_OBJECTS:asrobjs> )
target_link_libraries( asr PUBLIC
        -Wl,--no-as-needed open3d_torch_ops
        Open3D::Open3D
)
target_link_libraries( asr INTERFACE
        "${TORCH_LIBRARIES}"
        ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories( asr INTERFACE 
        "${CMAKE_CURRENT_DIRECTORY}" 
        "${CMAKE_CURRENT_BINARY_DIR}" 
        ${libcuckoo_INCLUDE_DIR}
        ${nanoflann_INCLUDE_DIR}
)

set_target_properties( asr PROPERTIES 
        INSTALL_RPATH $ORIGIN 
        BUILD_WITH_INSTALL_RPATH ON
)

install( TARGETS asr LIBRARY DESTINATION lib )
install( FILES 
        $<TARGET_FILE:open3d_torch_ops> 
        $<TARGET_FILE:Open3D::Open3D>
        $<TARGET_FILE:torch>
        $<TARGET_FILE:torch_cpu>
        $<TARGET_FILE:c10>
        ${torch_libgomp}
        DESTINATION lib 
)

# install resource files
file( GLOB_RECURSE asr_model_files ${PROJECT_SOURCE_DIR}/models/v*/model*.pt )
file( TOUCH dummy_resource_file )
list( APPEND asr_model_files dummy_resource_file )
install( FILES 
        ${asr_model_files}
        DESTINATION lib/asr_resources
)
