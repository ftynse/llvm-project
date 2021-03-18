//===- ParallelLoopDistrbute.cpp - Distribute loops around barriers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO(zinenko):
// Step 1: transform ifs to fors
// Step 2: assuming only fors can be nested inside parallel ops, add barriers
// around these fors
// Step 3: split pfors around barriers, cache any crossing use-def chains into
// alloca'ed memory (not optimal, but correct; scoping, storage reuse and
// duplicating ops with no side effects is kept for later.
// Repeat steps 2 and 3 until there are no more barriers and fors inside pfors
// (may be nested)

// Some barriers may be redundant, eliminate them in the same process.

#include "PassDetail.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "cpuify"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;

static void findValuesUsedBelow(Operation *op,
                                llvm::SetVector<Value> &crossing) {
  for (Operation *it = op->getPrevNode(); it != nullptr;
       it = it->getPrevNode()) {
    for (Value value : it->getResults()) {
      for (Operation *user : value.getUsers()) {
        // If the user is nested in another op, find its ancestor op that lives
        // in the same block as the barrier.
        while (user->getBlock() != op->getBlock())
          user = user->getBlock()->getParentOp();

        if (op->isBeforeInBlock(user)) {
          crossing.insert(value);
          break;
        }
      }
    }
  }

  // No need to process block arguments, they are assumed to be induction
  // variables and will be replicated.
}

static bool hasNestedBarrier(Operation *op) {
  auto result =
      op->walk([](scf::BarrierOp) { return WalkResult::interrupt(); });
  return result.wasInterrupted();
}

namespace {
struct ReplaceIfWithFors : public OpRewritePattern<scf::IfOp> {
  ReplaceIfWithFors(MLIRContext *ctx) : OpRewritePattern<scf::IfOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    assert(op.condition().getType().isInteger(1));

    // TODO: we can do this by having "undef" values as inputs, or do reg2mem.
    if (op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[if-to-for] 'if' with results, need reg2mem\n";
                 DBGS() << op);
      return failure();
    }

    if (!hasNestedBarrier(op)) {
      LLVM_DEBUG(DBGS() << "[if-to-for] no nested barrier\n");
      return failure();
    }

    Location loc = op.getLoc();
    auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<ConstantIndexOp>(loc, 1);
    auto cond = rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(),
                                             op.condition());
    auto thenLoop = rewriter.create<scf::ForOp>(loc, zero, cond, one);
    op->getParentOfType<FuncOp>()->dump();
    rewriter.mergeBlockBefore(op.getBody(0), &thenLoop.getBody()->back());
    rewriter.eraseOp(&thenLoop.getBody()->back());

    if (!op.elseRegion().empty()) {
      auto negCondition = rewriter.create<SubIOp>(loc, one, cond);
      auto elseLoop = rewriter.create<scf::ForOp>(loc, zero, negCondition, one);
      rewriter.mergeBlockBefore(op.getBody(1), &elseLoop.getBody()->back());
      rewriter.eraseOp(&elseLoop.getBody()->back());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct WrapForWithBarrier : public OpRewritePattern<scf::ForOp> {
  WrapForWithBarrier(MLIRContext *ctx) : OpRewritePattern<scf::ForOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    Operation *prevOp = op->getPrevNode();
    Operation *nextOp = op->getNextNode();

    if (!isa<scf::ParallelOp>(op->getParentOp())) {
      LLVM_DEBUG(DBGS() << "[wrap-for] not nested in a pfor\n");
      return failure();
    }

    if (!hasNestedBarrier(op)) {
      LLVM_DEBUG(DBGS() << "[wrap-for] no nested barrier\n");
      return failure();
    }

    bool hasPrevBarrierLike = prevOp == nullptr || isa<scf::BarrierOp>(prevOp);
    bool hasNextBarrierLike =
        nextOp == &op->getBlock()->back() || isa<scf::BarrierOp>(nextOp);
    if (hasPrevBarrierLike && hasNextBarrierLike) {
      LLVM_DEBUG(DBGS() << "[wrap-for] already has sufficient barriers\n");
      return failure();
    }

    if (!hasPrevBarrierLike)
      rewriter.create<scf::BarrierOp>(op.getLoc());

    if (!hasNextBarrierLike) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(op);
      rewriter.create<scf::BarrierOp>(op.getLoc());
    }

    // We don't actually change the op, but the pattern infra wants us to. Just
    // pretend we changed it in-place.
    rewriter.updateRootInPlace(op, [] {});

    return success();
  }
};

struct InterchangeForPFor : public OpRewritePattern<scf::ParallelOp> {
  InterchangeForPFor(MLIRContext *ctx)
      : OpRewritePattern<scf::ParallelOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    // A perfect nest must have two operations in the outermost body: a "for"
    // loop, and a terminator.
    if (std::next(op.getBody()->begin(), 2) != op.getBody()->end() ||
        !isa<scf::ForOp>(op.getBody()->front())) {
      LLVM_DEBUG(DBGS() << "[interchange] not a perfect pfor(for) nest\n");
      return failure();
    }

    // We shouldn't have parallel reduction loops coming from GPU anyway, and
    // sequential reduction loops can be transformed by reg2mem.
    auto forLoop = cast<scf::ForOp>(op.getBody()->front());
    if (op.getNumResults() != 0 || forLoop.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[interchange] not matching reduction loops\n");
      return failure();
    }

    if (!hasNestedBarrier(forLoop)) {
      LLVM_DEBUG(DBGS() << "[interchange] no nested barrier\n";);
    }

    auto newForLoop =
        rewriter.create<scf::ForOp>(forLoop.getLoc(), forLoop.lowerBound(),
                                    forLoop.upperBound(), forLoop.step());

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(newForLoop.getBody());
      auto newParallel = rewriter.create<scf::ParallelOp>(
          op.getLoc(), op.lowerBound(), op.upperBound(), op.step());
      // Merge in two stages so we can properly replace uses of two induction
      // varibales defined in different blocks.
      rewriter.mergeBlockBefore(op.getBody(), &newParallel.getBody()->back(),
                                newParallel.getInductionVars());
      rewriter.eraseOp(&newParallel.getBody()->back());
      rewriter.mergeBlockBefore(forLoop.getBody(),
                                &newParallel.getBody()->back(),
                                newForLoop.getInductionVar());
      rewriter.eraseOp(&newParallel.getBody()->back());
      rewriter.eraseOp(op);
    }
    return success();
  }
};

static std::pair<Block *, Block::iterator> getInsertionPointAfterDef(Value v) {
  if (Operation *op = v.getDefiningOp())
    return {op->getBlock(), std::next(Block::iterator(op))};

  BlockArgument blockArg = v.cast<BlockArgument>();
  return {blockArg.getParentBlock(), blockArg.getParentBlock()->begin()};
}

static std::pair<Block *, Block::iterator>
findNearestPostDominatingInsertionPoint(
    const std::pair<Block *, Block::iterator> &first,
    const std::pair<Block *, Block::iterator> &second,
    const PostDominanceInfo &postDominanceInfo) {
  // Same block, take the last op.
  if (first.first == second.first)
    return std::distance(first.second, second.second) < 0 ? first : second;

  // Same region, use "normal" dominance analysis.
  if (first.first->getParent() == second.first->getParent()) {
    // TODO: does this work with _post_-domination?
    Block *block =
        postDominanceInfo.findNearestCommonDominator(first.first, second.first);
    assert(block);
    if (block == first.first)
      return first;
    if (block == second.first)
      return second;
    return {block, block->begin()};
  }

  if (first.first->getParent()->isAncestor(second.first->getParent()))
    return second;

  assert(second.first->getParent()->isAncestor(first.first->getParent()) &&
         "expected values to be defined in nested regions");
  return first;
}

static std::pair<Block *, Block::iterator>
findNesrestPostDominatingInsertionPoint(
    ArrayRef<Value> values, const PostDominanceInfo &postDominanceInfo) {
  assert(!values.empty());
  std::pair<Block *, Block::iterator> insertPoint =
      getInsertionPointAfterDef(values[0]);
  for (unsigned i = 1, e = values.size(); i < e; ++i)
    insertPoint = findNearestPostDominatingInsertionPoint(
        insertPoint, getInsertionPointAfterDef(values[i]), postDominanceInfo);
  return insertPoint;
}

struct DistributeAroundBarrier : public OpRewritePattern<scf::ParallelOp> {
  DistributeAroundBarrier(MLIRContext *ctx)
      : OpRewritePattern<scf::ParallelOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[distribute] not matching reduction loops\n");
      return failure();
    }

    auto it =
        llvm::find_if(op.getBody()->getOperations(), [](Operation &nested) {
          return isa<scf::BarrierOp>(nested);
        });
    if (it == op.getBody()->end()) {
      LLVM_DEBUG(DBGS() << "[distribute] no barrier in the loop");
      return failure();
    }

    llvm::SetVector<Value> crossing;
    findValuesUsedBelow(&*it, crossing);

    // Find the earliest insertion point where loop bounds are fully defined.
    PostDominanceInfo postDominanceInfo(op->getParentOfType<FuncOp>());
    SmallVector<Value> operands;
    llvm::append_range(operands, op.lowerBound());
    llvm::append_range(operands, op.upperBound());
    llvm::append_range(operands, op.step());
    std::pair<Block *, Block::iterator> insertPoint =
        findNesrestPostDominatingInsertionPoint(operands, postDominanceInfo);

    // Emit code computing the total number of iterations in the loop. We don't
    // need to linearize them since we can allocate an nD array instead.
    SmallVector<Value> iterationCounts;
    rewriter.setInsertionPoint(insertPoint.first, insertPoint.second);
    for (auto bounds : llvm::zip(op.lowerBound(), op.upperBound(), op.step())) {
      Value lowerBound = std::get<0>(bounds);
      Value upperBound = std::get<1>(bounds);
      Value step = std::get<2>(bounds);
      Value diff = rewriter.create<SubIOp>(op.getLoc(), upperBound, lowerBound);
      Value count = rewriter.create<SignedCeilDivIOp>(op.getLoc(), diff, step);
      iterationCounts.push_back(count);
    }

    // Allocate space for values crossing the barrier.
    SmallVector<Value> allocations;
    allocations.reserve(crossing.size());
    SmallVector<int64_t> bufferSize(iterationCounts.size(),
                                    ShapedType::kDynamicSize);
    for (Value v : crossing) {
      auto type = MemRefType::get(bufferSize, v.getType());
      Value alloc =
          rewriter.create<memref::AllocaOp>(op.getLoc(), type, iterationCounts);
      allocations.push_back(alloc);
    }

    // Store values crossing the barrier in caches just before the barrier.
    rewriter.setInsertionPoint(&*it);
    for (auto pair : llvm::zip(crossing, allocations)) {
      Value v = std::get<0>(pair);
      Value alloc = std::get<1>(pair);
      rewriter.create<memref::StoreOp>(v.getLoc(), v, alloc,
                                       op.getInductionVars());
    }

    // Insert the terminator for the new loop immediately before the barrier.
    rewriter.create<scf::YieldOp>(op.getBody()->back().getLoc());

    // Create the second loop.
    rewriter.setInsertionPointAfter(op);
    auto newLoop = rewriter.create<scf::ParallelOp>(
        op.getLoc(), op.lowerBound(), op.upperBound(), op.step());
    rewriter.eraseOp(&newLoop.getBody()->back());

    // Note: this remapping makes this pattern incompatible with dialect
    // conversion. But there is no easy way around.
    rewriter.setInsertionPointToStart(newLoop.getBody());
    BlockAndValueMapping mapping;
    mapping.map(op.getInductionVars(), newLoop.getInductionVars());

    // Load back the cached values at the start of the second loop and make them
    // available for use through the mapping.
    for (auto pair : llvm::zip(crossing, allocations)) {
      Value orig = std::get<0>(pair);
      Value alloc = std::get<1>(pair);
      Value v = rewriter.create<memref::LoadOp>(orig.getLoc(), alloc,
                                                newLoop.getInductionVars());
      mapping.map(orig, v);
    }

    // Recreate the operations in the new loop with new values.
    SmallVector<Operation *> toDelete;
    toDelete.push_back(&*it);
    for (Operation *o = it->getNextNode(); o != nullptr; o = o->getNextNode()) {
      rewriter.clone(*o, mapping);
      toDelete.push_back(o);
    }

    // Erase original operations and the barrier.
    for (Operation *o : llvm::reverse(toDelete))
      rewriter.eraseOp(o);

    return success();
  }
};

struct CPUifyPass : public SCFCPUifyBase<CPUifyPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<ReplaceIfWithFors, WrapForWithBarrier, InterchangeForPFor>(
        &getContext());
    patterns.insert<DistributeAroundBarrier>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getFunction(), std::move(patterns),
                                            /*maxIterations=*/42)))
      signalPassFailure();
  }

  // void getDependentDialects(DialectRegistry &registry) const override {
  //   registry.insert<memref::MemRefDialect, StandardOpsDialect>();
  // }
};

} // end namespace

namespace mlir {
std::unique_ptr<Pass> createCPUifyPass() {
  return std::make_unique<CPUifyPass>();
}
} // namespace mlir
