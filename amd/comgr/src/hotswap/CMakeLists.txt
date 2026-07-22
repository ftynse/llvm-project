cmake_minimum_required(VERSION 3.20)
project(hotswap-transpiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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
# `target_link_libraries(amd_comgr PRIVATE hotswap::transpiler)` runs (gated
# on COMGR_ENABLE_HOTSWAP_TRANSPILE in the parent CMakeLists.txt). This puts
# the hotswap raiser and the rest of comgr in the same TU set so hotswap
# files can call comgr-metadata helpers directly without a layering inversion
# or a separate static archive.
add_library(hotswap-transpiler OBJECT
  raiser.cpp
  code-object-utils.cpp
  hotswap-error.cpp
  mc-state.cpp
  canonical-op.cpp
  opcode-map.cpp
  decode.cpp
)

if(NOT TARGET hotswap::transpiler)
  add_library(hotswap::transpiler ALIAS hotswap-transpiler)
endif()

# Match LLVM's compile flags (no-rtti, no-exceptions) — every LLVM type we
# consume relies on LLVM's hand-rolled RTTI rather than C++ typeid; mixing
# RTTI-on TUs with RTTI-off LLVM libs produces undefined-reference errors
# for `typeinfo` symbols on any class with a virtual method.
llvm_update_compile_flags(hotswap-transpiler)

set_target_properties(hotswap-transpiler PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Public include root so consumers can `#include "hotswap/raiser.h"`; the
# `..` also puts comgr's own src/ on the search path so code-object-utils.cpp
# can reach comgr-metadata.h / comgr-symbol.h.
target_include_directories(hotswap-transpiler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

# The decoder consumes AMDGPU target-private headers (SIDefines.h,
# SIInstrInfo.h, AMDGPUBaseInfo.h), which transitively include the
# TableGen-generated AMDGPUGen*.inc files. Those live in the LLVM source and
# build trees and are not exposed by an install tree.
# comgr-metadata.h -> comgr.h -> amd_comgr.h, configured into comgr's binary
# include/ directory (src/hotswap -> src -> comgr binary root).
get_filename_component(HOTSWAP_COMGR_BINARY_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/../.." ABSOLUTE)
target_include_directories(hotswap-transpiler PRIVATE
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU"
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU/Utils"
  "${LLVM_BINARY_DIR}/lib/Target/AMDGPU"
  "${HOTSWAP_COMGR_BINARY_DIR}/include"
)

target_link_libraries(hotswap-transpiler
  PUBLIC
    LLVMAMDGPUDesc
    LLVMAMDGPUDisassembler
    LLVMAMDGPUInfo
    LLVMAMDGPUUtils
    LLVMBinaryFormat
    LLVMCore
    LLVMMC
    LLVMMCDisassembler
    LLVMMCParser
    LLVMObject
    LLVMSupport
    LLVMTargetParser
)
