#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_SCFHOISTALLOCS
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

static void hoistAllocas(Operation *op) {
  // TODO: consider using a rewriter.
  op->walk([&](memref::AllocaOp alloca) {
    auto parent = alloca->getParentOfType<scf::ForOp>();
    if (!parent)
      return;
    while (auto grandParent = parent->getParentOfType<scf::ForOp>())
      parent = grandParent;

    alloca->moveBefore(parent);
  });
}

static void moveToStack(Operation *op, const DataLayoutAnalysis &dla,
                        int64_t maxSize) {
  SmallVector<std::pair<memref::AllocOp, memref::DeallocOp>> allocPairs;
  op->walk([&](memref::AllocOp alloc) {
    for (Operation &candidate :
         llvm::make_range(alloc->getIterator(), alloc->getBlock()->end())) {
      auto dealloc = dyn_cast<memref::DeallocOp>(candidate);
      if (!dealloc || dealloc.getMemref() != alloc.getMemref())
        continue;

      auto type = alloc.getMemref().getType();
      if (!type.hasStaticShape())
        continue;

      const DataLayout &dl = dla.getAtOrAbove(op);
      int64_t elementSize = dl.getTypeSize(type.getElementType());
      if (type.getNumElements() * elementSize >= maxSize)
        continue;

      allocPairs.emplace_back(alloc, dealloc);
    }
  });

  IRRewriter rewriter(op->getContext());
  for (auto &&[alloc, dealloc] : allocPairs) {
    rewriter.setInsertionPoint(alloc);
    rewriter.replaceOpWithNewOp<memref::AllocaOp>(
        alloc, alloc.getMemref().getType(), alloc.getOperands());
    rewriter.eraseOp(dealloc);
  }
}

namespace {
class HoistAllocsPass : public impl::SCFHoistAllocsBase<HoistAllocsPass> {
public:
  void runOnOperation() override {
    moveToStack(getOperation(), getAnalysis<DataLayoutAnalysis>(), 2 << 13);
    hoistAllocas(getOperation());
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createHoistAllocsPass() {
  return std::make_unique<HoistAllocsPass>();
}
