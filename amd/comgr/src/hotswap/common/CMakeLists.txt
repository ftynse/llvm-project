cmake_minimum_required(VERSION 3.20)
project(hotswap-common LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LLVM ---------------------------------------------------------------------
if(NOT TARGET LLVMSupport)
  find_package(LLVM REQUIRED CONFIG)
endif()
if(NOT COMMAND llvm_update_compile_flags)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
endif()

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# --- Library ------------------------------------------------------------------
# Path-agnostic pieces shared across the transpile path. Most of common/ is
# header-only (kernel-meta.h); hotswap-error carries one out-of-line symbol
# (HotswapError::ID), so it is compiled into a small OBJECT library both the
# loader and the decoder consume, keeping the error type in one place rather
# than coupling those directories to each other.
add_library(hotswap-common OBJECT
  hotswap-error.cpp
)

if(NOT TARGET hotswap::common)
  add_library(hotswap::common ALIAS hotswap-common)
endif()

llvm_update_compile_flags(hotswap-common)
# llvm_update_compile_flags does not reliably carry LLVM's -fno-rtti to a
# standalone find_package(LLVM) consumer, and HotswapError derives from
# llvm::ErrorInfo. Mirror LLVM_ENABLE_RTTI explicitly so we do not emit typeinfo
# an RTTI-off LLVM cannot resolve at link time.
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(hotswap-common PRIVATE /GR-)
  else()
    target_compile_options(hotswap-common PRIVATE -fno-rtti)
  endif()
endif()
set_target_properties(hotswap-common PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Public include root (src/) so consumers can `#include "hotswap/common/...".
target_include_directories(hotswap-common PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

target_link_libraries(hotswap-common
  PUBLIC
    LLVMSupport
)
