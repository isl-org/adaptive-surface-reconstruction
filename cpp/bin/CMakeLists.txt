file( GLOB asr_SOURCES  *.cpp *.cc *.c )

add_executable( asrtool ${asr_SOURCES} )

# enable c++14
set_target_properties( asrtool PROPERTIES 
                CXX_STANDARD 14
                CXX_STANDARD_REQUIRED ON 
                CXX_EXTENSIONS OFF)

# target_compile_options( asrtool PRIVATE -Wall -Wextra -Werror )

target_link_libraries( asrtool
        asr
        )

set_target_properties( asrtool PROPERTIES INSTALL_RPATH $ORIGIN/../lib )

install( TARGETS asrtool RUNTIME )