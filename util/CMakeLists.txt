add_library(util
    util.cpp
    data_type.cpp
    frustum.cpp
    file_mapping.cpp
    plane.cpp
    box.cpp
    tinyxml2.cpp)

set_target_properties(util PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_include_directories(util PUBLIC
	$<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

target_link_libraries(util PUBLIC TBB::tbb Threads::Threads)

target_compile_options(util PUBLIC
    $<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX2,-march=core-avx2>)

