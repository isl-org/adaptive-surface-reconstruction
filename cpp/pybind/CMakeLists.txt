
file( GLOB asrpybind_SOURCES  *.cpp *.cc *.c )

pybind11_add_module( asrpybind ${asrpybind_SOURCES} )

set_target_properties( asrpybind PROPERTIES 
        INSTALL_RPATH $ORIGIN 
        BUILD_WITH_INSTALL_RPATH ON 
)

target_include_directories( asrpybind PRIVATE )

target_link_libraries( asrpybind PRIVATE asr)