#ifndef MLIR_CONVERSION_STANDARDTOLLVM_LLVMTOLLVM_H_
#define MLIR_CONVERSION_STANDARDTOLLVM_LLVMTOLLVM_H_

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

std::unique_ptr<OperationPass<ModuleOp>> createLLVMToLLVMPass();
} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_LLVMTOLLVM_H_
