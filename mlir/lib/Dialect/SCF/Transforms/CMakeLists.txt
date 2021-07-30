add_mlir_dialect_library(MLIRSCFTransforms
  BarrierRemovalContinuation.cpp
  BarrierUtils.cpp
  Bufferize.cpp
  LoopPipelining.cpp
  LoopRangeFolding.cpp
  LoopSpecialization.cpp
  ParallelLoopDistribution.cpp
  ParallelLoopFusion.cpp
  ParallelLoopTiling.cpp
  StructuralTypeConversions.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SCF

  DEPENDS
  MLIRSCFPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRIR
  MLIRLLVMIR
  MLIRSCFToStandard
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)
