cmake_minimum_required(VERSION 3.20)
project(hotswap-raiser LANGUAGES CXX)

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
# OBJECT library so its translation units land directly in amd_comgr.so when
# `target_link_libraries(amd_comgr PRIVATE hotswap::raiser)` runs (gated
# on COMGR_ENABLE_HOTSWAP_TRANSPILE in the parent CMakeLists.txt). This puts
# the hotswap raiser and the rest of comgr in the same TU set so hotswap
# files can call comgr-metadata helpers directly without a layering inversion
# or a separate static archive.
add_library(hotswap-raiser OBJECT
  raiser.cpp
  raise_failure.cpp
  reg-file.cpp
  wave-projection.cpp
)

if(NOT TARGET hotswap::raiser)
  add_library(hotswap::raiser ALIAS hotswap-raiser)
endif()

# Match LLVM's compile flags (no-rtti, no-exceptions) — every LLVM type we
# consume relies on LLVM's hand-rolled RTTI rather than C++ typeid; mixing
# RTTI-on TUs with RTTI-off LLVM libs produces undefined-reference errors
# for `typeinfo` symbols on any class with a virtual method.
llvm_update_compile_flags(hotswap-raiser)

set_target_properties(hotswap-raiser PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Public include root (src/) so consumers can `#include "hotswap/raiser/raiser.h"`.
target_include_directories(hotswap-raiser PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

# The wave projection lowers decoded instructions to IR, so it links the
# decoder OBJECT library and inherits its curated LLVM component set (the
# AMDGPU MC/Desc/Utils libraries the projection's register queries need).
target_link_libraries(hotswap-raiser
  PUBLIC
    hotswap::decoder
    LLVMCore
    LLVMSupport
    LLVMTargetParser
)

# The wave projection consumes AMDGPU target-private headers (AMDGPUMCTargetDesc.h,
# AMDGPUBaseInfo.h), which transitively include the TableGen-generated
# AMDGPUGen*.inc files. Those live in the LLVM source and build trees and are not
# exposed by an install tree. Locate the source across the layouts comgr builds
# in: an in-tree monorepo build exposes it as LLVM_MAIN_SRC_DIR, a
# find_package(LLVM) consumer as LLVM_BUILD_MAIN_SRC_DIR, and the ROCm superbuild
# sets neither but keeps amd/comgr next to llvm/ under a shared root. Probe for
# the header rather than trust one variable, so a wrong or empty root fails the
# configure loudly.
set(HOTSWAP_AMDGPU_SRC_DIR "")
foreach(HotswapLLVMRoot
    "${LLVM_MAIN_SRC_DIR}"
    "${LLVM_BUILD_MAIN_SRC_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../llvm")
  if(HotswapLLVMRoot AND
     EXISTS "${HotswapLLVMRoot}/lib/Target/AMDGPU/SIDefines.h")
    get_filename_component(HOTSWAP_AMDGPU_SRC_DIR
      "${HotswapLLVMRoot}/lib/Target/AMDGPU" ABSOLUTE)
    break()
  endif()
endforeach()
if(NOT HOTSWAP_AMDGPU_SRC_DIR)
  message(FATAL_ERROR
    "hotswap-raiser: cannot find the LLVM AMDGPU target sources "
    "(SIDefines.h); set LLVM_MAIN_SRC_DIR to the llvm source directory.")
endif()
target_include_directories(hotswap-raiser PRIVATE
  "${HOTSWAP_AMDGPU_SRC_DIR}"
  "${HOTSWAP_AMDGPU_SRC_DIR}/Utils"
  "${LLVM_BINARY_DIR}/lib/Target/AMDGPU"
)
