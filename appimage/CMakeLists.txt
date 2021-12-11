
set( appdir "${CMAKE_CURRENT_BINARY_DIR}/AppDir" )
file( GLOB_RECURSE asr_resources_files 
        ${PROJECT_SOURCE_DIR}/models/v*/model*.pt 
        ${PROJECT_SOURCE_DIR}/datasets/*/*_attribution.txt
        )
list( APPEND asr_resources_files dummy_resource_file )

find_program(linuxdeploy_binary linuxdeploy-x86_64.AppImage )
if( NOT linuxdeploy_binary )
message( FATAL_ERROR "Building the appimage requires 'linuxdeploy-x86_64.AppImage' from https://github.com/linuxdeploy/linuxdeploy/releases/tag/continuous")
endif()

add_custom_target( appimage ALL
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${appdir}"
        COMMAND ${CMAKE_COMMAND} -E touch dummy_resource_file
        COMMAND ${CMAKE_COMMAND} -E make_directory "${appdir}/usr/lib/asr_resources"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${asr_resources_files} "${appdir}/usr/lib/asr_resources"
        COMMAND LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/../_deps/libtorch-src/lib VERSION=${PROJECT_VERSION} ${linuxdeploy_binary} --create-desktop-file --appdir "${appdir}" -e $<TARGET_FILE:asrtool> -i ${CMAKE_CURRENT_SOURCE_DIR}/asrtool.png -o appimage
        DEPENDS asrtool
)
