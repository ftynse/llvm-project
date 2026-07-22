function(canonicalize_cmake_boolean var)
    if(${var})
        set(${var} 1 PARENT_SCOPE)
    else()
        set(${var} 0 PARENT_SCOPE)
    endif()
endfunction()

canonicalize_cmake_boolean(COMGR_SPIRV_BACKEND_AVAILABLE)
canonicalize_cmake_boolean(COMGR_SPIRV_TRANSLATOR_AVAILABLE)

if ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
    set(COMGR_AMDGPU_TARGET_AVAILABLE 1)
else()
    set(COMGR_AMDGPU_TARGET_AVAILABLE 0)
endif()
canonicalize_cmake_boolean(COMGR_AMDGPU_TARGET_AVAILABLE)

canonicalize_cmake_boolean(COMGR_ENABLE_HOTSWAP_TRANSPILE)

find_program(LLVM_LIT_PATH
  NAMES llvm-lit llvm-lit.py llvm-lit.cmd
  PATHS "${LLVM_TOOLS_BINARY_DIR}/../../bin" "${LLVM_TOOLS_BINARY_DIR}"
  NO_DEFAULT_PATH
)
if (NOT LLVM_LIT_PATH)
  set(LLVM_LIT_PATH "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")
endif()
get_filename_component(COMGR_LIT_LLVM_TOOLS_DIR "${LLVM_LIT_PATH}" DIRECTORY)
message("-- LLVM_LIT_PATH: ${LLVM_LIT_PATH}")

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_custom_target(test-lit COMMAND "${LLVM_LIT_PATH}"
                  "${CMAKE_CURRENT_BINARY_DIR}" -v)

macro(add_comgr_lit_binary name lang)
  add_executable("${name}" "comgr-sources/${name}.${lang}")
  if (${lang} STREQUAL "c")
    set_target_properties("${name}" PROPERTIES
      C_STANDARD 99
      C_STANDARD_REQUIRED Yes
      C_EXTENSIONS No)
  endif()
  target_link_libraries("${name}" amd_comgr)
  add_dependencies(test-lit "${name}")
endmacro()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_comgr_lit_binary(source-to-bc-with-dev-libs c)
add_comgr_lit_binary(spirv-translator c)
add_comgr_lit_binary(compile-opencl-minimal c)
add_comgr_lit_binary(logger-redirect c)
add_comgr_lit_binary(compile-hip-minimal c)
add_comgr_lit_binary(compile-hip-asan c)
add_comgr_lit_binary(compile-hip-cxx-headers c)
add_comgr_lit_binary(spirv-to-reloc c)
add_comgr_lit_binary(source-to-spirv c)
add_comgr_lit_binary(unbundle c)
add_comgr_lit_binary(unpackage c)
add_comgr_lit_binary(get-version c)
add_comgr_lit_binary(status-string c)
add_comgr_lit_binary(data-action c)
add_comgr_lit_binary(lookup-code-object c)
add_comgr_lit_binary(hotswap-rewrite c)
add_comgr_lit_binary(parse-isa-name c)
add_comgr_lit_binary(name-expression-map c)

# The hotswap raiser driver used by the hotswap-raise lit tests. It links the
# hotswap-transpiler OBJECT library directly (not the amd_comgr C API) and
# recompiles the two comgr helpers whose internal COMGR:: symbols the shared
# library hides (getElfIsaName / getMetadataRoot / lookupSymbolByName), letting
# --gc-sections drop the rest of those translation units.
if(COMGR_ENABLE_HOTSWAP_TRANSPILE)
  add_executable(raise_cli
    comgr-sources/raise_cli.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/comgr-metadata.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/comgr-symbol.cpp)
  llvm_update_compile_flags(raise_cli)
  # LLVM is built without RTTI/exceptions; match it so cl::opt<std::string>'s
  # OptionValue typeinfo isn't referenced against the -fno-rtti Support library.
  target_compile_options(raise_cli PRIVATE
    -fno-rtti -fno-exceptions -ffunction-sections -fdata-sections)
  target_include_directories(raise_cli PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../include)
  target_link_libraries(raise_cli PRIVATE hotswap::transpiler)
  target_link_options(raise_cli PRIVATE -Wl,--gc-sections)
  add_dependencies(test-lit raise_cli)
endif()

add_dependencies(check-comgr test-lit)
