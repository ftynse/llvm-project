//===- BarrierUtil.h - Utilities for barrier removal --------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_SCF_TRANSFORMS_BARRIERUTILS_H_
#define MLIR_LIB_DIALECT_SCF_TRANSFORMS_BARRIERUTILS_H_

#include "mlir/IR/Block.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class OpBuilder;
class Value;
class ValueRange;

namespace scf {
class BarrierOp;
class ParallelOp;
} // namespace scf
} // namespace mlir

void findValuesUsedBelow(mlir::scf::BarrierOp barrier,
                         llvm::SetVector<mlir::Value> &crossing);

std::pair<mlir::Block *, mlir::Block::iterator>
findInsertionPointAfterLoopOperands(mlir::scf::ParallelOp op);

llvm::SmallVector<mlir::Value> emitIterationCounts(mlir::OpBuilder &builder,
                                                   mlir::scf::ParallelOp op);

mlir::Value allocateTemporaryBuffer(mlir::OpBuilder &builder, mlir::Value value,
                                    mlir::ValueRange iterationCounts);

#endif // MLIR_LIB_DIALECT_SCF_TRANSFORMS_BARRIERUTILS_H_
