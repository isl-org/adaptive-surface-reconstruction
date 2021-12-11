
set( python_package_out_dir "${CMAKE_CURRENT_BINARY_DIR}/python_package" )
file( GLOB_RECURSE asr_resources_files 
        ${PROJECT_SOURCE_DIR}/models/v*/model*.pt 
        ${PROJECT_SOURCE_DIR}/datasets/*/*_attribution.txt
        )
list( APPEND asr_resources_files dummy_resource_file )


# delete old wheels to make sure that there is only one
add_custom_target( python_wheel ALL
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${python_package_out_dir}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${python_package_out_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/adaptivesurfacereconstruction" "${python_package_out_dir}/adaptivesurfacereconstruction"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${python_package_out_dir}/adaptivesurfacereconstruction/asr_resources"
        COMMAND ${CMAKE_COMMAND} -E touch dummy_resource_file
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${asr_resources_files} "${python_package_out_dir}/adaptivesurfacereconstruction/asr_resources"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/setup.py" "${python_package_out_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/MANIFEST.in" "${python_package_out_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:asrpybind> "${python_package_out_dir}/adaptivesurfacereconstruction/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:asr> "${python_package_out_dir}/adaptivesurfacereconstruction/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Open3D::Open3D> "${python_package_out_dir}/adaptivesurfacereconstruction/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:open3d_torch_ops> "${python_package_out_dir}/adaptivesurfacereconstruction/"
        COMMAND ${patchelf_binary} --set-rpath $ORIGIN "${python_package_out_dir}/adaptivesurfacereconstruction/$<TARGET_FILE_NAME:open3d_torch_ops>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:torch> "${python_package_out_dir}/adaptivesurfacereconstruction/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:torch_cpu> "${python_package_out_dir}/adaptivesurfacereconstruction/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:c10> "${python_package_out_dir}/adaptivesurfacereconstruction/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${torch_libgomp} "${python_package_out_dir}/adaptivesurfacereconstruction/"
        COMMAND VERSION=${PROJECT_VERSION} ${CMAKE_COMMAND} -E chdir "${python_package_out_dir}" ${PYTHON_EXECUTABLE} setup.py bdist_wheel 
        DEPENDS asrpybind 
        VERBATIM
)

add_custom_target( install-pip-package
  COMMAND ${PYTHON_EXECUTABLE} -m pip install --upgrade --force-reinstall --no-index --find-links "." adaptivesurfacereconstruction
        DEPENDS python_wheel
        WORKING_DIRECTORY "${python_package_out_dir}/dist"
)

