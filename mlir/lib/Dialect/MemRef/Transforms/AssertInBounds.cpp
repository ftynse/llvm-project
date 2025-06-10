#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "assert-in-bounds"
#define DBGS() llvm::dbgs() << DEBUG_TYPE

namespace mlir::memref {
#define GEN_PASS_DEF_ASSERTINBOUNDSPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

// TODO: this does not belong here!
#define GEN_PASS_DEF_CHECKSTATICASSERTIONSPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
#define GEN_PASS_DEF_TESTINUSEANALYSISPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace mlir::memref

using namespace mlir;

/// Lattice state codes, mutually exclusive.
enum class InUseLatticeState : uint8_t {
  Uninitialized,
  Used,
  Unused,
  Undecidable
};

/// Storage for the lattice value, the structure is a canonical diamond:
///
/// ```
///       Undecidable
///        /        \
///     Used        Unused
///        \        /
///      Uninitialized
/// ```
class InUseLatticeStorage {
public:
  /// Lattice values are created in the bottom state.
  InUseLatticeStorage() : state(InUseLatticeState::Uninitialized) {}
  InUseLatticeStorage(const InUseLatticeState &state) : state(state) {}

  /// Joins two lattice values, potentially moving towards the top of the
  /// lattice. Returns the new value.
  static InUseLatticeStorage join(const InUseLatticeStorage &lhs,
                                  const InUseLatticeStorage &rhs) {
    if (lhs.state == rhs.state)
      return lhs.state;

    if (lhs.state == InUseLatticeState::Uninitialized)
      return rhs.state;

    if (rhs.state == InUseLatticeState::Uninitialized)
      return lhs.state;

    return InUseLatticeState::Undecidable;
  }

  /// Compares two lattice values for equality.
  bool operator==(const InUseLatticeStorage &other) const {
    return state == other.state;
  }

  /// DANGEROUS direct setter of the internal lattice state. This does not
  /// guarantee monotonicity, and MUST NOT be used inside dataflow propagation
  /// as its convergence is provided by monotonicity. Only for use by experts as
  /// a last resort.
  void unsafeSet(const InUseLatticeState &state) { this->state = state; }

  /// Prints the lattice value into the provided stream.
  // TODO: surface into the upstream documentation that this is mandatory.
  raw_ostream &print(raw_ostream &os) const {
    switch (state) {
    case InUseLatticeState::Uninitialized:
      return os << "uninitialized";
    case InUseLatticeState::Undecidable:
      return os << "undecidable";
    case InUseLatticeState::Unused:
      return os << "unused";
    case InUseLatticeState::Used:
      return os << "used";
    }
    llvm_unreachable("unsupport enum case");
  }

  /// Returns the lattice bottom value.
  static InUseLatticeStorage bottom() {
    return InUseLatticeState::Uninitialized;
  }

  /// Returns the lattice top value.
  static InUseLatticeStorage top() { return InUseLatticeState::Undecidable; }

  /// Returns the lattice enum state.
  const InUseLatticeState &get() const { return state; }

private:
  InUseLatticeState state;
};

/// The lattice indicating whether a value is used, e.g. for address computation
/// or control flow. The lattice itself does not indicate that the value is used
/// for.
class InUseLattice : public dataflow::Lattice<InUseLatticeStorage> {
public:
  using Lattice::Lattice;
};

/// Dataflow analysis indicating whether a value is used in address computation
/// or in control flow, and therefore must be present in the speculative
/// out-of-bounds checker. Propagates such known uses backwards in the dataflow
/// sense.
class InUseAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<InUseLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  /// This is called when an unanalyzable situation happens (and may also be
  /// called during initialization, bizarrely). Set the lattice to the top
  /// state to indicate the unanalyzable state.
  void setToExitState(InUseLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(InUseLatticeStorage::top()));
  }

  /// Directly override the great-grandparent visitation function to graciously
  /// handle the non-interprocedural mode: operands of the function return are
  /// considered unused for control flow instead of defaulting to unanalyzable.
  /// This is impossible to express otherwise. All other cases are deferred to
  /// `visitOperation` and friends below using the default logic.
  LogicalResult visit(ProgramPoint *point) override {
    bool needsSpecialHandling = [&] {
      if (point->isBlockStart())
        return false;
      Operation *op = point->getPrevOp();
      if (!op->hasTrait<OpTrait::ReturnLike>())
        return false;
      return isa_and_nonnull<FunctionOpInterface>(op->getParentOp());
    }();
    if (!needsSpecialHandling || getSolverConfig().isInterprocedural())
      return dataflow::AbstractSparseBackwardDataFlowAnalysis::visit(point);

    for (Value value : point->getPrevOp()->getOperands()) {
      InUseLattice *lattice = getLatticeElement(value);
      addDependency(lattice, point);
      propagateIfChanged(lattice, lattice->join(InUseLatticeState::Unused));
    }
    return success();
  }

  /// Directly override the great-grandparent initialization function to
  /// graciously handle the non-interprocedural mode: operands of the function
  /// return are considered unused for control flow instead of defaulting to
  /// unanalyzable. This is impossible to express otherwise.
  LogicalResult initialize(Operation *top) override {
    if (failed(
            dataflow::AbstractSparseBackwardDataFlowAnalysis::initialize(top)))
      return failure();

    if (getSolverConfig().isInterprocedural())
      return success();

    top->walk([this](Operation *op) {
      if (!op->hasTrait<OpTrait::ReturnLike>())
        return;
      if (!isa<FunctionOpInterface>(op->getParentOp()))
        return;
      for (Value operand : op->getOperands()) {
        InUseLattice *lattice = getLatticeElement(operand);
        lattice->getValue().unsafeSet(InUseLatticeState::Unused);
        addDependency(lattice, getProgramPointAfter(op));
        propagateIfChanged(lattice, ChangeResult::Change);
      }
    });
    return success();
  }

  /// Main visitation function, updates operand lattices based on result
  /// lattices and operation-specific information.
  LogicalResult
  visitOperation(Operation *op, ArrayRef<InUseLattice *> operands,
                 ArrayRef<const InUseLattice *> results) override {

    // Mark operands coming from a specific ODS group as used/unused. This is
    // hacky, but avoids introducing an interface.
    auto markODSGroupsAs = [&](auto op, ArrayRef<unsigned> groups,
                               InUseLatticeState state) {
      for (unsigned group : groups) {
        auto [firstIndex, indexLength] = op.getODSOperandIndexAndLength(group);
        for (unsigned i = 0; i < indexLength; ++i) {
          InUseLattice *lattice = operands[firstIndex + i];
          propagateIfChanged(lattice, lattice->join(state));
        }
      }
    };

    llvm::TypeSwitch<Operation *>(op)
        // Both load base and indices are used.
        .Case<memref::LoadOp, vector::LoadOp>([&](auto loadOp) {
          markODSGroupsAs(loadOp, {0, 1}, InUseLatticeState::Used);
        })
        // Stored values are considered unused, the rest is used.
        .Case<memref::StoreOp, vector::StoreOp>([&](auto storeOp) {
          markODSGroupsAs(storeOp, {0}, InUseLatticeState::Unused);
          markODSGroupsAs(storeOp, {1, 2}, InUseLatticeState::Used);
        })
        .Default([&](Operation *op) {
          // Be conservative for everything non-pure.
          if (!isPure(op)) {
            for (InUseLattice *operandLattice : operands) {
              propagateIfChanged(
                  operandLattice,
                  operandLattice->join(InUseLatticeStorage::top()));
            }
            return;
          }
          // Pure operations just propagate.
          for (InUseLattice *operandLattice : operands) {
            ChangeResult change = ChangeResult::NoChange;
            for (const InUseLattice *resultLattice : results) {
              change |= operandLattice->join(*resultLattice);
            }
            propagateIfChanged(operandLattice, change);
          }
        });
    return success();
  }

  /// Calls are not supported yet.
  void visitCallOperand(OpOperand &) override { assert(false && "no calls"); }

  /// Visitation function along branching and region control flow.
  void visitBranchOperand(OpOperand &opOperand) override {
    InUseLattice *lattice = getLatticeElement(opOperand.get());
    // Non-forwarded operands of if's and for's are used in their control flow.
    llvm::TypeSwitch<Operation *>(opOperand.getOwner())
        .Case<scf::IfOp, scf::ForOp>([&](auto) {
          propagateIfChanged(lattice, lattice->join(InUseLatticeState::Used));
        })
        .Default([&](Operation *) {
          propagateIfChanged(lattice,
                             lattice->join(InUseLatticeStorage::top()));
        });
  }
};

/// Attach in-use information as attributes to operations defining values
/// (results and block arguments).
static LogicalResult testInUseAnalysis(Operation *top) {
  DataFlowConfig config;
  config.setInterprocedural(false);
  DataFlowSolver solver(config);
  SymbolTableCollection symbolTable;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<InUseAnalysis>(symbolTable);
  if (failed(solver.initializeAndRun(top)))
    return top->emitError() << "failed to complete in-use analysis";

  // Pre-create attributes so we don't take the lock all the time during the
  // walk.
  auto latticeBottom = StringAttr::get(top->getContext(), "bottom");
  auto used = StringAttr::get(top->getContext(), "used");
  auto unused = StringAttr::get(top->getContext(), "unused");
  auto latticeTop = StringAttr::get(top->getContext(), "top");
  auto missing = StringAttr::get(top->getContext(), "missing");

  auto getAttrForValueList = [&](ValueRange values) -> Attribute {
    return ArrayAttr::get(
        top->getContext(),
        llvm::map_to_vector(values, [&](Value result) -> Attribute {
          auto *inUseLattice = solver.lookupState<InUseLattice>(result);
          if (!inUseLattice)
            return missing;

          switch (inUseLattice->getValue().get()) {
          case InUseLatticeState::Uninitialized:
            return latticeBottom;
          case InUseLatticeState::Unused:
            return unused;
          case InUseLatticeState::Used:
            return used;
          case InUseLatticeState::Undecidable:
            return latticeTop;
          }
          llvm_unreachable("unhandled enum case");
        }));
  };

  top->walk([&](Operation *op) {
    if (isa<ModuleOp>(op))
      return;
    op->setAttr("__in_use_results", getAttrForValueList(op->getResults()));
    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
      for (auto &&[j, block] : llvm::enumerate(op->getRegion(i))) {
        op->setAttr("__in_use_blockarg_" + std::to_string(i) + "_" +
                        std::to_string(j),
                    getAttrForValueList(block.getArguments()));
      }
    }
  });
  return success();
}

/// Kind of checks to insert.
enum class CheckKind { PerDimension, Combined };

// These should become an interface eventually, but currently not worth the
// complexity of adding one.

/// Whether the operation is supposed by the assertion inserter.
static bool isSupported(memref::LoadOp) { return true; }
static bool isSupported(memref::StoreOp) { return true; }
static bool isSupported(vector::LoadOp loadOp) {
  return !loadOp.getVectorType().isScalable();
}
static bool isSupported(vector::StoreOp storeOp) {
  return !storeOp.getVectorType().isScalable();
}

/// Returns the number of elements accessed along the given dimension by a
/// vector load/store operation.
template <typename OpTy>
static int64_t getAccessExtentAlongDimVectorOp(OpTy op, unsigned dim) {
  static_assert(llvm::is_one_of<OpTy, vector::LoadOp, vector::StoreOp>::value,
                "expected vector load or store");
  if (isa<VectorType>(op.getMemRefType().getElementType()))
    return 1;
  unsigned leadingOneDims =
      op.getMemRefType().getRank() - op.getVectorType().getRank();
  return dim < leadingOneDims
             ? 1
             : op.getVectorType().getDimSize(dim - leadingOneDims);
}

/// Returns the number of elements accessed along the given dimension by the
/// operation.
static int64_t getAccessExtentAlongDim(memref::LoadOp, unsigned) { return 1; }
static int64_t getAccessExtentAlongDim(memref::StoreOp, unsigned) { return 1; }
static int64_t getAccessExtentAlongDim(vector::LoadOp loadOp, unsigned dim) {
  return getAccessExtentAlongDimVectorOp(loadOp, dim);
}
static int64_t getAccessExtentAlongDim(vector::StoreOp storeOp, unsigned dim) {
  return getAccessExtentAlongDimVectorOp(storeOp, dim);
}

/// Returns the base memref that is being indexed into by the accessing
/// operation.
static Value getAccessBase(memref::LoadOp loadOp) { return loadOp.getMemRef(); }
static Value getAccessBase(memref::StoreOp storeOp) {
  return storeOp.getMemRef();
}
static Value getAccessBase(vector::LoadOp loadOp) { return loadOp.getBase(); }
static Value getAccessBase(vector::StoreOp storeOp) {
  return storeOp.getBase();
}

// End pseudo-interface.

/// Inserts `cf.assert` checking whether the subscripts of the given
/// memory-accessing operation are in bounds.
template <typename OpTy>
static LogicalResult insertInBoundsAssertions(OpBuilder &builder, OpTy op,
                                              CheckKind checkKind) {
  if (!isSupported(op))
    return op.emitError() << "unsupported variation of the op";

  ImplicitLocOpBuilder b(op->getLoc(), builder);
  Value zero = b.createOrFold<arith::ConstantIndexOp>(0);
  Value totalCheck =
      checkKind == CheckKind::Combined
          ? b.createOrFold<arith::ConstantIntOp>(1, b.getI1Type())
          : nullptr;
  for (unsigned i = 0, e = op.getMemRefType().getRank(); i < e; ++i) {
    Value index = b.createOrFold<arith::ConstantIndexOp>(i);
    Value dim = b.createOrFold<memref::DimOp>(getAccessBase(op), index);
    Value subscript = op.getIndices()[i];
    Value lowerBoundCheck = b.createOrFold<arith::CmpIOp>(
        arith::CmpIPredicate::sge, subscript, zero);

    int64_t accessExtent = getAccessExtentAlongDim(op, i);
    assert(accessExtent >= 1 && "expected positive access extent");
    Value lastAccessedIndex =
        accessExtent == 1
            ? subscript
            : b.createOrFold<arith::AddIOp>(
                  subscript,
                  b.createOrFold<arith::ConstantIndexOp>(accessExtent - 1));

    Value upperBoundCheck = b.createOrFold<arith::CmpIOp>(
        arith::CmpIPredicate::slt, lastAccessedIndex, dim);
    Value boundCheck =
        b.createOrFold<arith::AndIOp>(lowerBoundCheck, upperBoundCheck);
    if (checkKind == CheckKind::PerDimension) {
      b.createOrFold<cf::AssertOp>(
          boundCheck,
          "memref access out of bounds along dimension " + std::to_string(i));
    } else {
      assert(checkKind == CheckKind::Combined && "unsupported check kind");
      totalCheck = b.createOrFold<arith::AndIOp>(totalCheck, boundCheck);
    }
  }
  if (checkKind == CheckKind::Combined) {
    b.createOrFold<cf::AssertOp>(totalCheck, "memref access out of bounds");
  }
  return success();
}

namespace {
/// Options for the in-bound assertion inserter.
struct InsertInBoundsAssertionsConfig {
  CheckKind checkKind;
  bool includeVectorLoadStore;
  bool warnOnUnknown;
};
} // namespace

/// Inserts in-bounds check for the given operation immediately prior to it.
/// Specific checks depend on the kind of operation and the configuration. Emits
/// errors when checks cannot be inserted and warnings, if requested in the
/// configuration, on unhandled operations that may need checks.
static LogicalResult
insertInBoundsAssertionDispatch(OpBuilder &builder, Operation *op,
                                const InsertInBoundsAssertionsConfig &config) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<memref::LoadOp, memref::StoreOp>([&](auto casted) {
        return insertInBoundsAssertions(builder, casted, config.checkKind);
      })
      .Case<vector::LoadOp, vector::StoreOp>([&](auto casted) {
        // Vector load/store specifically allow for lowering-defined
        // out-of-bounds access when using scalar-typed memory. Ignore
        // those unless explicitly requested by the caller.
        if (!config.includeVectorLoadStore &&
            !isa<VectorType>(casted.getMemRefType().getElementType()))
          return success();

        return insertInBoundsAssertions(builder, casted, config.checkKind);
      })
      .Default([&](Operation *uncasted) {
        if (!config.warnOnUnknown)
          return success();

        auto effecting = dyn_cast<MemoryEffectOpInterface>(uncasted);
        if (!effecting)
          return success();

        SmallVector<MemoryEffects::EffectInstance> effects;
        effecting.getEffects(effects);
        if (llvm::none_of(effects, [](MemoryEffects::EffectInstance &instance) {
              bool effectMayBeOnMemRef =
                  !instance.getValue() ||
                  isa<MemRefType>(instance.getValue().getType());
              return effectMayBeOnMemRef &&
                     isa<MemoryEffects::Read, MemoryEffects::Write>(
                         instance.getEffect());
            }))
          return success();

        uncasted->emitWarning()
            << "operation with memory effects was not processed";
        return success();
      });
}

// TODO: these should be part of the function op interface. Note the caveat for
// the dependent dialects of the pass using it if the call does not belong to
// the same dialect as the func op (very unlikely).

/// Creates a call to the given function with the provided operands. The
/// operands are expected to be compatible with the function. The builder must
/// have an appropriate insertion point set up.
static Operation *createCall(OpBuilder &builder, Location loc,
                             FunctionOpInterface func, ValueRange operands) {
  if (auto funcFunc = dyn_cast<func::FuncOp>(*func)) {
    return builder.create<func::CallOp>(loc, funcFunc, operands);
  }
  func.emitError() << "cannot create a call to this function";
  return nullptr;
}

/// Creates a return from the given function with the provided operands. The
/// operands are expected to be compatible with the function results. The
/// builder must have an appropriate insertion point set up.
static Operation *createReturn(OpBuilder &builder, Location loc,
                               FunctionOpInterface func, ValueRange operands) {
  if (auto funcFunc = dyn_cast<func::FuncOp>(*func)) {
    return builder.create<func::ReturnOp>(loc, operands);
  }
  func.emitError() << "cannot create a return from this function";
  return nullptr;
}

/// Creates a dummy value, typically zero, to be used as operand that is not
/// participating in control flow or indexing, e.g., secondary loop iterators
/// that may be needed for structural correctness while cloning operations but
/// don't affect the logic.
static Value createDummyValue(OpBuilder &builder, Location loc, Type type) {
  if (type.isInteger()) {
    return builder.create<arith::ConstantOp>(loc, type,
                                             builder.getIntegerAttr(type, 0));
  }
  if (type.isIndex()) {
    return builder.create<arith::ConstantIndexOp>(loc, 0);
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return builder.create<arith::ConstantFloatOp>(
        loc, llvm::APFloat::getZero(floatType.getFloatSemantics()), floatType);
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    Value element = createDummyValue(builder, loc, vecType.getElementType());
    if (!element)
      return nullptr;
    return builder.create<vector::SplatOp>(loc, element, vecType);
  }
  return nullptr;
}

/// Indicates which operations will be replaced with an in-bounds check.
static bool isReplacedWithCheck(Operation *op) {
  return isa<memref::LoadOp, memref::StoreOp, vector::LoadOp, vector::StoreOp>(
      op);
}

/// Creates a new function performing a speculative check that all known kinds
/// of memory accesses in the given function are in bounds, and inserts a call
/// to this function as the first operation in the given function. Reports
/// errors and returns failure if such a check cannot be created, in particular
/// if it would require speculating operations that should not be, such as
/// stores to memory.
static LogicalResult
createSpeculativeFunction(RewriterBase &rewriter, FunctionOpInterface funcOp,
                          const DataFlowSolver &solver,
                          const InsertInBoundsAssertionsConfig &config) {
  IRMapping mapping;
  OpBuilder::InsertionGuard guard(rewriter);

  // Values that are known-used or could not be analyzed are considered as
  // potentially used.
  auto mayBeUsed = [&](Value value) {
    auto *lattice = solver.lookupState<InUseLattice>(value);
    if (!lattice) {
      LLVM_DEBUG(DBGS() << "no lattice for " << value
                        << ", presuming dead code");
      return false;
    }
    return lattice->getValue().get() == InUseLatticeState::Used ||
           lattice->getValue().get() == InUseLatticeState::Undecidable;
  };

  // Collect operations that should be cloned into the speculative functions.
  DenseSet<Operation *> activeOperations;
  activeOperations.insert(funcOp);
  WalkResult walkResult = funcOp->walk([&](Operation *op) {
    // Be eager about control flow for now, we use direct block mapping to find
    // insertion points below.
    if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>()) {
      activeOperations.insert(op);
      return WalkResult::advance();
    }

    if (!llvm::any_of(op->getResults(), mayBeUsed))
      return WalkResult::advance();

    // Don't speculate operations that shouldn't be.
    // TODO: we may need additional dependency analysis to figure out if we can
    // speculate reads, i.e., as long we as are not reading values that were
    // stored before and reads are not volatile, we should be fine.
    if (!isSpeculatable(op)) {
      op->emitError() << "in-bounds check generation requires speculating this "
                         "operation, but it is not speculatable";
      return WalkResult::interrupt();
    }
    activeOperations.insert(op);
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  auto isActive = [&](Operation *op) { return activeOperations.contains(op); };

  // Returns `true` if the given operation is a terminator returning from the
  // function being processed.
  auto isFuncReturn = [&](Operation *op) {
    return op->hasTrait<OpTrait::IsTerminator>() &&
           op->getNumSuccessors() == 0 && op->getParentOp() == funcOp;
  };

  // Must walk in pre-order to ensure regions/blocks are created before
  // entering them.
  walkResult = funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isActive(op) && !isReplacedWithCheck(op))
      return WalkResult::advance();

    // When about to clone an op, create dummy values for anything that is not
    // active, e.g., secondary iteration arguments that are not participating in
    // control flow. This will preserve syntactic correctness without the
    // runtime overhead of actually computing those values.
    for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
      if (!mayBeUsed(operand) && !mapping.contains(operand)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(
            mapping.lookup(operand.getParentBlock()));
        Value dummy =
            createDummyValue(rewriter, op->getLoc(), operand.getType());
        if (!dummy) {
          op->emitError() << "could not create a speculative operand #" << i;
          return WalkResult::interrupt();
        }
        mapping.map(operand, dummy);
      }
    }

    if (op == funcOp) {
      rewriter.setInsertionPoint(funcOp);
      // Block *entryBlock = clonedFuncOp.addEntryBlock();
      // mapping.map(&funcOp.getFunctionBody().front(), entryBlock);
      // mapping.map(funcOp.getArguments(), entryBlock->getArguments());
      // return WalkResult::advance();
    } else {
      rewriter.setInsertionPointToEnd(mapping.lookup(op->getBlock()));
    }

    // Insert the checks where required. Note that insertion operates around an
    // existing memory-accessing operation, so we first clone it, call the
    // insertion, and then erase the cloned operation.
    if (isReplacedWithCheck(op)) {
      Operation *cloned = rewriter.clone(*op, mapping);
      if (failed(insertInBoundsAssertionDispatch(rewriter, cloned, config)))
        return WalkResult::advance();
      rewriter.eraseOp(cloned);
      for (Value result : op->getResults())
        mapping.erase(result);
      return WalkResult::advance();
    }

    // Create explicit no-operand returns from the function since we removed all
    // results.
    if (isFuncReturn(op)) {
      if (createReturn(rewriter, op->getLoc(), funcOp, /*operands=*/{}) ==
          nullptr)
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    // Otherwise clone the operation. Create blocks for regions when
    // appropriate, but don't clone the regions themselves. They will be
    // populated later.
    Operation *cloned = rewriter.cloneWithoutRegions(*op, mapping);
    for (auto &&[origRegion, cloneRegion] :
         llvm::zip_equal(op->getRegions(), cloned->getRegions())) {
      for (Block &block : origRegion.getBlocks()) {
        SmallVector<Location> argLocs =
            llvm::map_to_vector(block.getArguments(),
                                [](BlockArgument arg) { return arg.getLoc(); });
        Block *clonedBlock = rewriter.createBlock(
            &cloneRegion, {}, block.getArgumentTypes(), argLocs);
        mapping.map(&block, clonedBlock);
        mapping.map(block.getArguments(), clonedBlock->getArguments());
      }
    }

    // For the function itself, adapt its signature to not return anything.
    if (op == funcOp) {
      auto clonedFuncOp = cast<FunctionOpInterface>(cloned);
      clonedFuncOp.setName("__speculative_in_bounds_check_" +
                           clonedFuncOp.getName().str());
      clonedFuncOp.setVisibility(SymbolTable::Visibility::Private);
      if (failed(clonedFuncOp.eraseResults(
              llvm::BitVector(funcOp.getNumResults(), true)))) {
        funcOp->emitError() << "cannot create a speculative checker function "
                               "with no results to "
                               "match this function";
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  // Call the speculation function from the original function.
  auto clonedFuncOp = cast<FunctionOpInterface>(mapping.lookup(funcOp));
  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
  Operation *call = createCall(rewriter, clonedFuncOp->getLoc(), clonedFuncOp,
                               funcOp.getArguments());
  return success(call != nullptr);
}

/// Inserts bounds check assertions in place.
static LogicalResult
insertInPlaceInBoundsAssertions(Operation *root,
                                const InsertInBoundsAssertionsConfig &config) {
  OpBuilder builder(root->getContext());
  WalkResult walkResult = root->walk([&](Operation *op) {
    OpBuilder::InsertionGuard raii(builder);
    builder.setInsertionPoint(op);

    if (failed(insertInBoundsAssertionDispatch(builder, op, config)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

/// For all functions in the given scope, creates new functions speculatively
/// checking bounds for all known memory-accessing operations, and inserts calls
/// to those functions.
static LogicalResult insertSpeculativeInBoundsAssertions(
    Operation *root, const InsertInBoundsAssertionsConfig &config) {
  DataFlowConfig dataFlowConfig;
  dataFlowConfig.setInterprocedural(false);
  DataFlowSolver solver(dataFlowConfig);
  SymbolTableCollection symbolTable;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<InUseAnalysis>(symbolTable);
  if (failed(solver.initializeAndRun(root)))
    return root->emitError() << "failed to complete in-use analysis";

  IRRewriter rewriter(root->getContext());
  WalkResult walkResult =
      root->walk<WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
        // if (failed(createSpeculativeInBoundsChecks(rewriter, funcOp,
        // config)))
        //   return WalkResult::interrupt();
        if (failed(createSpeculativeFunction(rewriter, funcOp, solver, config)))
          return WalkResult::interrupt();
        return WalkResult::skip();
      });
  return failure(walkResult.wasInterrupted());
}

namespace {
class AssertInBoundsPass
    : public memref::impl::AssertInBoundsPassBase<AssertInBoundsPass> {
public:
  using AssertInBoundsPassBase::AssertInBoundsPassBase;

  void runOnOperation() override;
};

void AssertInBoundsPass::runOnOperation() {
  InsertInBoundsAssertionsConfig config;
  config.checkKind =
      checkEachDim ? CheckKind::PerDimension : CheckKind::Combined;
  config.includeVectorLoadStore = includeVectorLoadStore;
  config.warnOnUnknown = warnOnUnknown;

  if (!createSpeculativeFuncs) {
    if (failed(insertInPlaceInBoundsAssertions(getOperation(), config)))
      signalPassFailure();
    return;
  }

  if (failed(insertSpeculativeInBoundsAssertions(getOperation(), config)))
    signalPassFailure();
}

class CheckStaticAssertionsPass
    : public memref::impl::CheckStaticAssertionsPassBase<
          CheckStaticAssertionsPass> {
public:
  void runOnOperation() override {
    WalkResult walkResult = getOperation()->walk([&](cf::AssertOp assertOp) {
      APInt value;
      if (matchPattern(assertOp.getArg(), m_ConstantInt(&value)) &&
          value.isZero()) {
        assertOp->emitError() << "assertion known to be false";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};

class TestInUseAnalysisPass
    : public memref::impl::TestInUseAnalysisPassBase<TestInUseAnalysisPass> {
public:
  void runOnOperation() override {
    if (failed(testInUseAnalysis(getOperation())))
      return signalPassFailure();
  }
};
} // namespace
