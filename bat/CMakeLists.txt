add_library(bat
    attribute.cpp
    ba_tree.cpp
    lba_tree_builder.cpp)

set_target_properties(bat PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_include_directories(bat PUBLIC
    $<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

target_link_libraries(bat PUBLIC TBB::tbb util)

