if (VexCL_FOUND)
    vexcl_add_executables(eigen_vexcl eigen_vexcl.cpp)
    target_link_libraries(eigen_vexcl INTERFACE amgcl)
endif()