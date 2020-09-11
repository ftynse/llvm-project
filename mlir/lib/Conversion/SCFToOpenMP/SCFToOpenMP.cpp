//===- SCFToOpenMP.cpp - Conversion from SCF loops to OpenMP --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "../PassDetail.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class ParallelOpConversion : public ConversionPattern {
public:
  ParallelOpConversion(FuncOp numThreadsFunc, FuncOp threadNumFunc)
      : ConversionPattern(scf::ParallelOp::getOperationName(), 1,
                          numThreadsFunc.getContext()),
        getNumThreadsFunc(numThreadsFunc), getThreadNumFunc(threadNumFunc) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loop = cast<scf::ParallelOp>(op);
    if (loop.getNumLoops() != 1)
      return rewriter.notifyMatchFailure(op, "only single loops supported");
    if (loop.getNumResults() != 0)
      return rewriter.notifyMatchFailure(op,
                                         "loops with reduction not supported");

    Location loc = op->getLoc();

    auto ompParallel = rewriter.create<omp::ParallelOp>(
        loc, nullptr, nullptr,
        rewriter.getStringAttr(
            omp::stringifyClauseDefault(omp::ClauseDefault::defshared)),
        ArrayRef<Value>(), ArrayRef<Value>(), ArrayRef<Value>(),
        ArrayRef<Value>(), nullptr);
    OpBuilder::InsertionGuard guard(rewriter);
    assert(ompParallel.region().empty());
    rewriter.createBlock(&ompParallel.region(), ompParallel.region().end());
    Value numThreads =
        rewriter.create<CallOp>(loc, getNumThreadsFunc).getResult(0);
    Value threadId =
        rewriter.create<CallOp>(loc, getThreadNumFunc).getResult(0);
    Value casted =
        rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(), numThreads);
    Value newStep = rewriter.create<MulIOp>(loc, loop.step()[0], casted);
    Value threadIdIndex =
        rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(), threadId);
    Value newLb = rewriter.create<MulIOp>(loc, loop.step()[0], threadIdIndex);
    rewriter.create<scf::ForOp>(
        loc, newLb, loop.upperBound()[0], newStep, llvm::None,
        [&](OpBuilder &nested, Location nestedLoc, Value iv, ValueRange) {
          BlockAndValueMapping mapping;
          mapping.map(loop.getInductionVars().front(), iv);
          for (Operation &op : loop.getBody()->without_terminator())
            rewriter.clone(op, mapping);
          nested.create<scf::YieldOp>(nestedLoc);
        });
    rewriter.create<omp::TerminatorOp>(loc);
    rewriter.eraseOp(op);

    return success();
  }

  FuncOp getNumThreadsFunc;
  FuncOp getThreadNumFunc;
};

class SCFToOpenMPPass : public ConvertSCFToOpenMPBase<SCFToOpenMPPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    if (op->getNumRegions() == 0)
      return;

    if (op->getNumRegions() != 1) {
      op->emitOpError() << "expected an op with 0 or 1 regions";
      return signalPassFailure();
    }

    if (!op->hasTrait<OpTrait::SymbolTable>()) {
      op->emitOpError() << "expected an op that has a symbol table";
      return signalPassFailure();
    }

    OpBuilder b(&getContext());
    b.setInsertionPointToStart(&op->getRegion(0).front());
    SymbolTable symbols(op);
    auto numThreadsFunc = symbols.lookup<FuncOp>("omp_get_num_threads");
    if (!numThreadsFunc)
      numThreadsFunc =
          b.create<FuncOp>(UnknownLoc::get(ctx), "omp_get_num_threads",
                           FunctionType::get(llvm::None, b.getI32Type(), ctx));
    auto threadNumFunc = symbols.lookup<FuncOp>("omp_get_thread_num");
    if (!threadNumFunc)
      threadNumFunc =
          b.create<FuncOp>(UnknownLoc::get(ctx), "omp_get_thread_num",
                           FunctionType::get(llvm::None, b.getI32Type(), ctx));

    OwningRewritePatternList patterns;
    patterns.insert<ParallelOpConversion>(numThreadsFunc, threadNumFunc);

    ConversionTarget target(*ctx);
    target.addLegalDialect<omp::OpenMPDialect>();
    target.addDynamicallyLegalOp<scf::ParallelOp>([](scf::ParallelOp op) {
      return op.getParentOfType<scf::ParallelOp>() ||
             op.getParentOfType<omp::ParallelOp>() || op.getNumResults() != 0;
    });
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<scf::ForOp, scf::YieldOp>();

    if (failed(applyPartialConversion(op, target, patterns)))
      signalPassFailure();
  }
};
} // end namespace

std::unique_ptr<Pass> mlir::createSCFToOpenMPPass() {
  return std::make_unique<SCFToOpenMPPass>();
}
