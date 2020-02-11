//===- MDH.cpp - Using Linalg transformations for MDH scheme --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

using namespace mlir;

/// Tiles `op` and inserts (promotion) copies to a local cache for all data
/// accessed unconditionally.
static linalg::LinalgOp
tileAndPromote(OpBuilder &builder, OperationFolder &folder, linalg::LinalgOp op,
               ArrayRef<Value> tileSizes, ArrayRef<unsigned> permutation) {
  // Preserve the state of the builder.
  OpBuilder::InsertionGuard raii(builder);
  builder.setInsertionPoint(op);

  // Perform loop tiling (blocking) with the specified sizes.
  Optional<linalg::TiledLinalgOp> tile =
      linalg::tileLinalgOp(builder, op, tileSizes, permutation, &folder);
  if (!tile)
    return nullptr;

  // Collect all subviews.
  auto innermostLoop = cast<loop::ForOp>(tile->loops.back());
  SmallVector<Value, 4> allSubviews;
  innermostLoop.walk([&allSubviews](SubViewOp subview) {
    allSubviews.push_back(subview.getResult());
  });

  // Make sure we insert promotions immediately above the main operation, which
  // is now wrapped in loops.
  builder.setInsertionPoint(tile->op);
  linalg::promoteSubViews(builder, op.getLoc(), allSubviews,
                          /*dynamicBuffers=*/false, &folder);
  op.erase();
  return tile->op;
}

/// Creates an operation with the given name that returns a single index
/// element. Returns the resulting value of this operation.
static Value createTunableOp(OpBuilder &builder, Location loc, StringRef name) {
  OperationState hack(loc, name);
  hack.addTypes(builder.getIndexType());
  return builder.createOperation(hack)->getResult(0);
}

/// Tiles and promotes twice, injecting tunable operations as sources of tile
/// sizes.
static LogicalResult doTransform(linalg::LinalgOp op) {
  // Set up the builder infrastructure.
  Location loc = op.getLoc();
  OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);
  edsc::ScopedContext context(builder, loc);
  OperationFolder folder(op.getContext());

  // Create dummy operations that produce tile sizes.
  Value ts0 = createTunableOp(builder, loc, "tunable.tile.size.0");
  Value ts1 = createTunableOp(builder, loc, "tunable.tile.size.1");
  Value ts2 = createTunableOp(builder, loc, "tunable.tile.size.2");
  Value ts3 = createTunableOp(builder, loc, "tunable.tile.size.3");
  Value ts4 = createTunableOp(builder, loc, "tunable.tile.size.4");
  Value ts5 = createTunableOp(builder, loc, "tunable.tile.size.5");

  // Tile and promote twice, using the results of dummy operations as tile
  // sizes.
  op = tileAndPromote(builder, folder, op, {ts0, ts1, ts2}, /*permutation=*/{});
  if (!op)
    return failure();

  op = tileAndPromote(builder, folder, op, {ts3, ts4, ts5}, /*permutation=*/{});
  if (!op)
    return failure();

  return success();
}

namespace {
/// Pass performing tiling and promotion on Linalg ops annotated with "mdh" unit
/// attribute.
struct MDHScheme : public FunctionPass<MDHScheme> {
  MDHScheme() = default;

  void runOnFunction() override {
    getFunction().walk([this](linalg::LinalgOp op) {
      if (!op.getAttrOfType<UnitAttr>("mdh"))
        return;
      if (failed(doTransform(op)))
        signalPassFailure();
    });
  }
};
} // namespace

/// Static registration of the pass with mlir-opt.
static PassRegistration<MDHScheme>
    registrationMDH("linalg-mdh", "Apply MDH scheme to Linalg operations",
                    [] { return std::make_unique<MDHScheme>(); });

/// Tile sizes to use instead of tunable operations.
llvm::cl::list<unsigned> clTileSizes("linalg-mdh-tile-sizes",
                                     llvm::cl::ZeroOrMore,
                                     llvm::cl::MiscFlags::CommaSeparated);

/// Replaces the given operation (which is expected to be tunable.tile.size.#
/// where # is a 0-based decimal integer) with a constant provided through the
/// command line. The CLI flag is expected to have a sufficient number of
/// values for all operations.
static LogicalResult replaceTunableParam(Operation *op) {
  // Slice the operation name to extract the integer part and check we have
  // a value provided in CLI for it.
  StringRef numStr = op->getName().getStringRef().drop_until(
      [](char c) { return c >= '0' && c <= '9'; });
  unsigned num;
  if (numStr.getAsInteger(/*Radix=*/10, num))
    return op->emitOpError("unsupported op: ") << op->getName().getStringRef();

  if (num >= clTileSizes.size())
    return op->emitOpError("no value provided for this tunable parameter");

  // Replace the operation with the constant operation defining the tile size.
  OpBuilder builder(op);
  Location loc = op->getLoc();
  Value cst = builder.create<ConstantIndexOp>(loc, clTileSizes[num]);
  op->getResult(0).replaceAllUsesWith(cst);
  op->erase();

  return success();
}

namespace {
/// Pass performing the replacement of tunable operations with constants
/// provided in the command line.
struct ReplaceTunables : public FunctionPass<ReplaceTunables> {
  ReplaceTunables() = default;

  void runOnFunction() override {
    getFunction().walk([this](Operation *op) {
      if (!op->getName().getStringRef().startswith("tunable.tile.size"))
        return;
      if (failed(replaceTunableParam(op)))
        signalPassFailure();
    });
  }
};
} // namespace

/// Static registration of the pass with mlir-opt.
static PassRegistration<ReplaceTunables>
    registrationReplace("linalg-mdh-apply-tuned",
                        "Subsitute tunable values with constants",
                        [] { return std::make_unique<ReplaceTunables>(); });
