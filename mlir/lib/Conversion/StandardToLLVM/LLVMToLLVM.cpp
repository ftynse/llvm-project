//===- LLVMToLLVM.cpp - Amends types in LLVM dialect ops ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/StandardToLLVM/LLVMToLLVM.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct LLVMOpLowering : public ConversionPattern {
  explicit LLVMOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, Pattern::MatchAnyOpTypeTag(), 1,
                          &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter *converter = getTypeConverter();
    SmallVector<Type> convertedResultTypes;
    if (failed(converter->convertTypes(op->getResultTypes(),
                                       convertedResultTypes)))
      return failure();
    if (convertedResultTypes == op->getResultTypes())
      return failure();

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(convertedResultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      state.addRegion();

    Operation *rewritten = rewriter.createOperation(state);
    rewriter.replaceOp(op, rewritten->getResults());

    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      rewriter.inlineRegionBefore(op->getRegion(i), rewritten->getRegion(i),
                                  rewritten->getRegion(i).begin());

    return success();
  }
};

struct LLVMToLLVMPass : public ConvertLLVMToLLVMBase<LLVMToLLVMPass> {
  LLVMToLLVMPass() = default;

  void runOnOperation() override {
    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<LLVMOpLowering>(typeConverter);
    ConversionTarget target(getContext());
    target.addDynamicallyLegalDialect<LLVM::LLVMDialect>(
        [&](Operation *op) -> Optional<bool> {
          SmallVector<Type> convertedResultTypes;
          if (failed(typeConverter.convertTypes(op->getResultTypes(),
                                                convertedResultTypes)))
            return llvm::None;
          return convertedResultTypes == op->getResultTypes();
        });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createLLVMToLLVMPass() {
  return std::make_unique<LLVMToLLVMPass>();
}
