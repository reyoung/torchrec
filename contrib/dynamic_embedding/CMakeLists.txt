cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)

project(torchrec_dynamic_embedding
        VERSION 0.0.1
        LANGUAGES CXX C)

find_package(Torch REQUIRED)
find_package(PythonExtensions REQUIRED)


if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(TDE_IS_TOP_LEVEL_PROJECT ON)
else()
    set(TDE_IS_TOP_LEVEL_PROJECT OFF)
endif()

option(TDE_WITH_TESTING "Enable unittest in C++ side" ${TDE_IS_TOP_LEVEL_PROJECT})

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    option(TDE_WITH_CXX11_ABI "GLIBCXX use c++11 ABI or not. libtorch installed by conda is not use it by default," OFF)
    if (NOT CMAKE_TOOLCHAIN_FILE) # if toolchain file not set.
        if (TDE_WITH_CXX11_ABI)
            set(CMAKE_TOOLCHAIN_FILE
                    "${CMAKE_CURRENT_SOURCE_DIR}/toolchains/gcc_use_cxx11_abi.cmake"
                CACHE FILEPATH "Default toolchain"
            )
        else()
            set(CMAKE_TOOLCHAIN_FILE
                    "${CMAKE_CURRENT_SOURCE_DIR}/toolchains/gcc_not_use_cxx11_abi.cmake"
                CACHE FILEPATH "Default toolchain"
            )
        endif()
    endif()
endif()

if (TDE_WITH_TESTING)
    enable_testing()
endif()
add_subdirectory(src)
