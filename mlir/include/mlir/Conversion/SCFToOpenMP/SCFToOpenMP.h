//===- ConvertSCFToStandard.h - Convert SCF Ops to OpenMP -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SCFTOOPENMP_SCFTOOPENMP_H
#define MLIR_CONVERSION_SCFTOOPENMP_SCFTOOPENMP_H

#include <memory>

namespace mlir {
class Pass;

/// Creates a pass to convert scf.parallel ops to omp.parallel containing a
/// sequential SCF loop.
std::unique_ptr<Pass> createSCFToOpenMPPass();
} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOOPENMP_SCFTOOPENMP_H
