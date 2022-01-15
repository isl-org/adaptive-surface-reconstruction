# Run this script from the repository root like this
#   cmake -P utils/update_boilerplate.cmake
#
# The script expects 4 files in the repository root
# - .boilerplate_cpp
# - .boilerplate_py
# - .boilerplate_cpp_old
# - .boilerplate_py_old
# with the old and new boilerplate for the respective file type.
#
set( BOILERPLATE_CPP ".boilerplate_cpp" )
set( BOILERPLATE_PY ".boilerplate_py" )
set( BOILERPLATE_CPP_OLD ".boilerplate_cpp_old" )
set( BOILERPLATE_PY_OLD ".boilerplate_py_old" )

foreach( x IN ITEMS ${BOILERPLATE_CPP} ${BOILERPLATE_PY} ${BOILERPLATE_CPP_OLD} ${BOILERPLATE_PY_OLD})
    if( NOT EXISTS "${x}" )
        message(FATAL_ERROR "file '${x}' does not exist")
    endif()
endforeach()

file(READ "${BOILERPLATE_CPP}" boilerplate_cpp )
file(READ "${BOILERPLATE_PY}" boilerplate_py )
file(READ "${BOILERPLATE_CPP_OLD}" boilerplate_cpp_old )
file(READ "${BOILERPLATE_PY_OLD}" boilerplate_py_old )

file( GLOB_RECURSE files cpp/*.cpp datasets/*.py models/*.py python/*.py )
foreach( filepath IN ITEMS ${files} )

    get_filename_component(ex "${filepath}" EXT)
    if( ex STREQUAL ".py")
        set( old_boilerplate "${boilerplate_py_old}" )
        set( new_boilerplate "${boilerplate_py}" )
    elseif( ex STREQUAL ".cpp")
        set( old_boilerplate "${boilerplate_cpp_old}" )
        set( new_boilerplate "${boilerplate_cpp}" )
    endif()

    file(READ "${filepath}" data )
    
    string(FIND "${data}" "${old_boilerplate}" pos )
    if( pos GREATER_EQUAL 0 )
        string(REPLACE "${old_boilerplate}" "${new_boilerplate}" new_data "${data}" )
        file(WRITE "${filepath}" "${new_data}" )
        message(STATUS "updating ${filepath}")
    else()
        string(FIND "${data}" "${new_boilerplate}" pos )
        if( pos GREATER_EQUAL 0 )
            message(STATUS "already updated: ${filepath}")
        else()
            message(STATUS "no match for ${filepath}")
        endif()
    endif()

endforeach()