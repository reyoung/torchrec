# Catch2 section.
if (TDE_WITH_TESTING)
    add_subdirectory(Catch2)
endif()

# tcb::span section. A std::span implementation in C++ 11.
# the tcb::span is enable_testing globally. so we do not add_subdirectory(span)
add_library(tcb_span INTERFACE)
target_include_directories(tcb_span INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/span/include)
add_library(tcb::span ALIAS tcb_span)
