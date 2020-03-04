add_mlir_dialect_library(MLIRAffine
  AffineOps.cpp
  AffineValueMap.cpp
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine
  )
add_dependencies(MLIRAffine

  MLIRAffineOpsIncGen
  MLIREDSC
  MLIRIR
  MLIRLoopLikeInterfaceIncGen
  MLIRStandardOps
  )
target_link_libraries(MLIRAffine

  MLIREDSC
  MLIRIR
  MLIRStandardOps
  )
