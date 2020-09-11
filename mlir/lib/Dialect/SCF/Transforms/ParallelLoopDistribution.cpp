//===- ParallelLoopDistrbute.cpp - Distribute loops around barriers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/// Finds all uses of value that are located after `op` in its containing block.
static void findUsesBelow(Value value, Operation *op,
                          SmallVectorImpl<OpOperand *> &users) {
  for (auto &u : value.getUses())
    if (op->isBeforeInBlock(u.getOwner()))
      users.push_back(&u);
}

/// Checks whether value has uses located after `op` in its containing block.
static bool hasUsesBelow(Value value, Operation *op) {
  SmallVector<OpOperand *, 4> users;
  findUsesBelow(value, op, users);
  return !users.empty();
}

/// Distributes a parallel loop that contains the given `barrier` into two
/// loops. The barrier remains in the first loop. Not compatible with the
/// pattern rewriter.
static LogicalResult distributeAround(scf::BarrierOp barrier,
                                      OpBuilder &builder) {
  auto loop = cast<scf::ParallelOp>(barrier.getParentOp());
  if (loop.getNumReductions() != 0)
    return failure();

  for (Operation *op = barrier.getOperation()->getPrevNode(); op;
       op = op->getPrevNode()) {
    // TODO: instead of bailing out, we can either replicate the slice if all
    // operations in it are without side effects or store the result in a
    // scratchpad and load it later.
    for (Value res : op->getResults())
      if (hasUsesBelow(res, barrier))
        return failure();
  }

  // Clone operations that follow the barrier into a new loop, rewriting any
  // uses of induction variables.
  Location terminatorLoc = loop.getBody()->getTerminator()->getLoc();
  SmallVector<Operation *, 8> toDelete;
  builder.setInsertionPointAfter(loop);
  builder.create<scf::ParallelOp>(
      loop.getLoc(), loop.lowerBound(), loop.upperBound(), loop.step(),
      llvm::None,
      [&](OpBuilder &nested, Location nloc, ValueRange iters, ValueRange) {
        BlockAndValueMapping mapping;
        mapping.map(loop.getInductionVars(), iters);
        for (Operation *op = barrier.getOperation()->getNextNode(); op;
             op = op->getNextNode()) {
          nested.clone(*op, mapping);
          toDelete.push_back(op);
        }
      });

  builder.setInsertionPointToEnd(loop.getBody());
  builder.create<scf::YieldOp>(terminatorLoc);
  for (Operation *op : llvm::reverse(toDelete))
    op->erase();

  return success();
}

namespace {
struct DistributionPass
    : public SCFParallelLoopDistributionBase<DistributionPass> {
  void runOnFunction() override {
    SmallVector<scf::ParallelOp, 8> loops;
    getFunction().walk(
        [&loops](scf::ParallelOp loop) { loops.push_back(loop); });

    while (!loops.empty()) {
      scf::ParallelOp loop = loops.pop_back_val();
      OpBuilder builder(loop);
      for (Operation &op : *loop.getBody()) {
        if (auto barrier = dyn_cast<scf::BarrierOp>(&op)) {
          if (failed(distributeAround(barrier, builder)))
            continue;
          loops.push_back(
              cast<scf::ParallelOp>(loop.getOperation()->getNextNode()));
          barrier.erase();
          break;
        }
      }
    }
  }
};
} // end namespace

std::unique_ptr<Pass> mlir::createParallelLoopDistributionPass() {
  return std::make_unique<DistributionPass>();
}
