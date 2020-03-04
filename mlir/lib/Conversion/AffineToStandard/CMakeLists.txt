add_mlir_conversion_library(MLIRAffineToStandard
  AffineToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AffineToStandard
)
add_dependencies(
  MLIRAffineToStandard

  MLIRAffine
  MLIRStandardOps
  MLIRIR
  LLVMCore
  LLVMSupport
)
target_link_libraries(
  MLIRAffineToStandard

  MLIRAffine
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  MLIRTransforms
  MLIRIR
  LLVMCore
  LLVMSupport
)
