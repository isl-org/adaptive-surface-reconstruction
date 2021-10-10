
set( appdir "${CMAKE_CURRENT_BINARY_DIR}/AppDir" )
file( GLOB_RECURSE asr_model_files ${PROJECT_SOURCE_DIR}/models/v*/model*.pt )
list( APPEND asr_model_files dummy_resource_file )

find_program(linuxdeploy_binary linuxdeploy-x86_64.AppImage)

add_custom_target( appimage
        COMMAND ${CMAKE_COMMAND} -E touch dummy_resource_file
        COMMAND ${CMAKE_COMMAND} -E make_directory "${appdir}/usr/lib/asr_resources"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${asr_model_files} "${appdir}/usr/lib/asr_resources"
        COMMAND LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/../_deps/libtorch-src/lib VERSION=${PROJECT_VERSION} ${linuxdeploy_binary} --create-desktop-file --appdir "${appdir}" -e $<TARGET_FILE:asrtool> -i ${CMAKE_CURRENT_SOURCE_DIR}/asrtool.png -o appimage
        DEPENDS asrtool
)
