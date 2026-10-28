#include "aiputile/Passes.h"
#include "aiputile/AIPUTileOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace mlir;
using aiputile::TileLoopOp;
using aiputile::TileType;

namespace {
TileType tileTypeForTensor(RankedTensorType tensor) {
  if (!tensor.hasStaticShape() || tensor.getEncoding() ||
      !tensor.getElementType().isF32())
    return {};
  int64_t n = 1;
  for (int64_t dim : tensor.getShape()) {
    if (dim <= 0 || dim > std::numeric_limits<int32_t>::max() / n)
      return {};
    n *= dim;
  }
  return TileType::get(tensor.getContext(), n, tensor.getElementType());
}

// Source op names are the existing registered AIPUGC operations in the full
// driver. The portable harness accepts their generic spelling independently
// of the proprietary graph-builder implementation.
struct ImportBinary : ConversionPattern {
  ImportBinary(StringRef source, TypeConverter &types, MLIRContext *ctx)
      : ConversionPattern(types, source, 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumRegions() ||
        failed(aiputile::verifySourceTensorTypes(op)))
      return failure();
    for (NamedAttribute attr : op->getAttrs()) {
      if (attr.getName() == "dictAttr") {
        if (auto dict = dyn_cast<DictionaryAttr>(attr.getValue());
            dict && dict.empty())
          continue;
      }
      if (attr.getName() == "with_activation") {
        if (auto value = dyn_cast<StringAttr>(attr.getValue());
            value && value.getValue() == "NONE")
          continue;
      }
      return op->emitError("tile path does not support source parameter ")
             << attr.getName();
    }
    Type result = getTypeConverter()->convertType(op->getResult(0).getType());
    if (!result)
      return failure();
    if (op->getName().getStringRef() == "aipugc.add")
      rewriter.replaceOpWithNewOp<aiputile::AddOp>(op, result, operands[0],
                                                   operands[1]);
    else
      rewriter.replaceOpWithNewOp<aiputile::MulOp>(op, result, operands[0],
                                                   operands[1]);
    return success();
  }
};

struct ImportFunction : ConversionPattern {
  ImportFunction(MLIRContext *ctx) : ConversionPattern("aipugc.func", 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value>,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op->getAttrOfType<TypeAttr>("function_type");
    auto name = op->getAttrOfType<StringAttr>("sym_name");
    if (!type || !isa<FunctionType>(type.getValue()) || !name ||
        op->getNumRegions() != 1)
      return op->emitError("expected a well-formed aipugc.func");
    auto func = rewriter.create<func::FuncOp>(
        op->getLoc(), name.getValue(), cast<FunctionType>(type.getValue()));
    func->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op->getRegion(0), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ImportReturn : ConversionPattern {
  ImportReturn(MLIRContext *ctx) : ConversionPattern("aipugc.return", 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> values;
    for (auto item : llvm::enumerate(operands)) {
      Value value = item.value();
      if (isa<TileType>(value.getType()))
        value = rewriter.create<aiputile::ToTensorOp>(
            op->getLoc(), op->getOperand(item.index()).getType(), value);
      values.push_back(value);
    }
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, values);
    return success();
  }
};

// Cancel conversion materializations between tile producers and consumers.
// Deliberately do not run a greedy DCE here: a second source consumer must be
// visible to fusion eligibility checks, even when its result is currently dead.
void cancelIntermediateBridges(ModuleOp module) {
  bool changed;
  do {
    changed = false;
    SmallVector<aiputile::FromTensorOp> bridges;
    module.walk([&](aiputile::FromTensorOp op) { bridges.push_back(op); });
    for (auto from : bridges) {
      auto to = from.getInput().getDefiningOp<aiputile::ToTensorOp>();
      if (!to || to.getInput().getType() != from.getOutput().getType())
        continue;
      from.getOutput().replaceAllUsesWith(to.getInput());
      from.erase();
      if (to.getOutput().use_empty())
        to.erase();
      changed = true;
    }
  } while (changed);
}

struct ImportPass : PassWrapper<ImportPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportPass)
  StringRef getArgument() const final { return "lower-aipugc-to-aiputile"; }
  StringRef getDescription() const final {
    return "Convert AIPUGC arithmetic to immutable tile values with tensor ABI "
           "bridges";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<aiputile::AIPUTileDialect, func::FuncDialect>();
  }
  void runOnOperation() override {
    // Diagnose unsupported source types before operand remapping can fail
    // generically while trying to materialize their Tile type.
    bool invalidSource = false;
    getOperation().walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (name == "aipugc.add" || name == "aipugc.mul") {
        if (op->getNumOperands() != 2 || op->getNumRegions() ||
            failed(aiputile::verifySourceTensorTypes(op)))
          invalidSource = true;
      }
    });
    if (invalidSource)
      return signalPassFailure();
    TypeConverter types;
    types.addConversion([](Type type) { return type; });
    types.addConversion(
        [](RankedTensorType type) -> Type { return tileTypeForTensor(type); });
    // Value-returning callbacks also work with SDKs that wrap materializations
    // in std::optional<Value>. Each direction has a single supported bridge.
    types.addTargetMaterialization([](OpBuilder &builder, TileType result,
                                      ValueRange inputs,
                                      Location loc) -> Value {
      if (inputs.size() != 1 || !isa<RankedTensorType>(inputs[0].getType()))
        return Value{};
      return builder.create<aiputile::FromTensorOp>(loc, result, inputs[0])
          .getOutput();
    });
    types.addSourceMaterialization(
        [](OpBuilder &builder, RankedTensorType result, ValueRange inputs,
           Location loc) -> Value {
          if (inputs.size() != 1 || !isa<TileType>(inputs[0].getType()))
            return Value{};
          return builder.create<aiputile::ToTensorOp>(loc, result, inputs[0])
              .getOutput();
        });
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return op->getName().getDialectNamespace() != "aipugc";
    });
    RewritePatternSet patterns(&getContext());
    patterns.add<ImportBinary>("aipugc.add", types, &getContext());
    patterns.add<ImportBinary>("aipugc.mul", types, &getContext());
    patterns.add<ImportFunction, ImportReturn>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      getOperation().emitError(
          "tile path supports only static dense FP32 Add/Mul and "
          "function/return; unsupported source IR remains");
      return signalPassFailure();
    }
    cancelIntermediateBridges(getOperation());
  }
};

struct TilingPass : PassWrapper<TilingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilingPass)
  TilingPass() = default;
  TilingPass(const TilingPass &other) : PassWrapper(other) {}
  explicit TilingPass(int64_t size) { tileSize = size; }
  Option<int64_t> tileSize{
      *this, "tile-size",
      llvm::cl::desc("Common partition capacity for all tile computations"),
      llvm::cl::init(256)};
  StringRef getArgument() const final { return "aiputile-tile"; }
  StringRef getDescription() const final {
    return "Partition each tile Add/Mul independently using a bounded symbolic "
           "tile_loop";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<aiputile::AIPUTileDialect>();
  }
  void runOnOperation() override {
    if (tileSize <= 0 || tileSize > std::numeric_limits<int32_t>::max()) {
      getOperation().emitError("tile-size must be in [1, INT32_MAX]");
      return signalPassFailure();
    }
    SmallVector<Operation *> binaries;
    getOperation().walk([&](Operation *op) {
      if (isa<aiputile::AddOp, aiputile::MulOp>(op) &&
          !op->getParentOfType<TileLoopOp>())
        binaries.push_back(op);
    });
    for (Operation *op : binaries) {
      auto full = cast<TileType>(op->getResult(0).getType());
      auto local = TileType::get(
          &getContext(), std::min<int64_t>(tileSize, full.getCapacity()),
          full.getElementType());
      OpBuilder builder(op);
      auto loop =
          builder.create<TileLoopOp>(op->getLoc(), full, op->getOperands(),
                                     builder.getI64IntegerAttr(tileSize));
      Block *body = new Block();
      loop.getBody().push_back(body);
      for (unsigned i = 0; i < op->getNumOperands(); ++i)
        body->addArgument(local, op->getLoc());
      body->addArgument(builder.getIndexType(), op->getLoc());
      body->addArgument(builder.getIndexType(), op->getLoc());
      OpBuilder inner = OpBuilder::atBlockEnd(body);
      Value computed;
      if (isa<aiputile::AddOp>(op))
        computed = inner.create<aiputile::AddOp>(
            op->getLoc(), local, body->getArgument(0), body->getArgument(1));
      else
        computed = inner.create<aiputile::MulOp>(
            op->getLoc(), local, body->getArgument(0), body->getArgument(1));
      inner.create<aiputile::YieldOp>(op->getLoc(), computed);
      op->getResult(0).replaceAllUsesWith(loop.getOutput());
      op->erase();
    }
  }
};

int64_t partitionSize(TileLoopOp loop) {
  return std::min<int64_t>(loop.getTileSize(),
                           loop.getOutput().getType().getCapacity());
}

// Merge equal-domain regions, retaining the producer computation as ordinary
// tile SSA in the consumer body. No specialized whole-tensor fused op is used.
void fuseLoops(TileLoopOp producer, TileLoopOp consumer) {
  SmallVector<Value> inputs;
  DenseMap<Value, unsigned> inputIndex;
  auto append = [&](Value value) {
    if (!inputIndex.count(value)) {
      inputIndex[value] = inputs.size();
      inputs.push_back(value);
    }
  };
  for (Value input : producer.getInputs())
    append(input);
  for (Value input : consumer.getInputs())
    if (input != producer.getOutput())
      append(input);
  OpBuilder builder(consumer);
  auto fused = builder.create<TileLoopOp>(consumer.getLoc(),
                                          consumer.getOutput().getType(),
                                          inputs, consumer.getTileSizeAttr());
  auto local = TileType::get(consumer.getContext(), partitionSize(consumer),
                             consumer.getOutput().getType().getElementType());
  Block *body = new Block();
  fused.getBody().push_back(body);
  for (Value input : inputs) {
    (void)input;
    body->addArgument(local, consumer.getLoc());
  }
  body->addArgument(builder.getIndexType(), consumer.getLoc());
  body->addArgument(builder.getIndexType(), consumer.getLoc());
  OpBuilder inner = OpBuilder::atBlockEnd(body);
  IRMapping pmap;
  for (auto entry : llvm::enumerate(producer.getInputs()))
    pmap.map(producer.getBody().front().getArgument(entry.index()),
             body->getArgument(inputIndex.lookup(entry.value())));
  pmap.map(producer.getBody().front().getArgument(producer.getInputs().size()),
           body->getArgument(inputs.size()));
  pmap.map(
      producer.getBody().front().getArgument(producer.getInputs().size() + 1),
      body->getArgument(inputs.size() + 1));
  for (Operation &op : producer.getBody().front().without_terminator())
    inner.clone(op, pmap);
  Value temporary = pmap.lookup(
      cast<aiputile::YieldOp>(producer.getBody().front().back()).getValue());
  IRMapping cmap;
  for (auto entry : llvm::enumerate(consumer.getInputs())) {
    Value value = entry.value() == producer.getOutput()
                      ? temporary
                      : body->getArgument(inputIndex.lookup(entry.value()));
    cmap.map(consumer.getBody().front().getArgument(entry.index()), value);
  }
  cmap.map(consumer.getBody().front().getArgument(consumer.getInputs().size()),
           body->getArgument(inputs.size()));
  cmap.map(
      consumer.getBody().front().getArgument(consumer.getInputs().size() + 1),
      body->getArgument(inputs.size() + 1));
  for (Operation &op : consumer.getBody().front().without_terminator())
    inner.clone(op, cmap);
  inner.create<aiputile::YieldOp>(
      consumer.getLoc(),
      cmap.lookup(cast<aiputile::YieldOp>(consumer.getBody().front().back())
                      .getValue()));
  consumer.getOutput().replaceAllUsesWith(fused.getOutput());
  consumer.erase();
  producer.erase();
}

struct FusionPass : PassWrapper<FusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionPass)
  StringRef getArgument() const final { return "aiputile-fuse-tiles"; }
  StringRef getDescription() const final {
    return "Fuse same-domain single-consumer tile_loop regions";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<aiputile::AIPUTileDialect>();
  }
  void runOnOperation() override {
    bool invalid = false;
    getOperation().walk([&](Operation *op) {
      if (isa<aiputile::AddOp, aiputile::MulOp>(op) &&
          !op->getParentOfType<TileLoopOp>()) {
        op->emitError("run aiputile-tile before tile fusion");
        invalid = true;
      }
    });
    // Preflight before mutating any region or discarding dead consumers.
    getOperation().walk([&](TileLoopOp producer) {
      for (Operation *user : producer.getOutput().getUsers()) {
        auto consumer = dyn_cast<TileLoopOp>(user);
        if (!consumer)
          continue;
        if (!producer.getOutput().hasOneUse()) {
          producer.emitError(
              "fusion requires a single-consumer tile_loop result");
          invalid = true;
        }
        if (producer->getBlock() != consumer->getBlock() ||
            producer.getOutput().getType() != consumer.getOutput().getType() ||
            partitionSize(producer) != partitionSize(consumer)) {
          consumer.emitError(
              "cannot fuse different tile domains, partitions or blocks");
          invalid = true;
        }
      }
    });
    if (invalid)
      return signalPassFailure();
    bool changed;
    do {
      changed = false;
      SmallVector<TileLoopOp> loops;
      getOperation().walk([&](TileLoopOp op) { loops.push_back(op); });
      for (auto consumer : loops) {
        TileLoopOp producer;
        for (Value value : consumer.getInputs())
          if ((producer = value.getDefiningOp<TileLoopOp>()))
            break;
        if (producer) {
          fuseLoops(producer, consumer);
          changed = true;
          break;
        }
      }
    } while (changed);
  }
};

LogicalResult checkKernel(func::FuncOp func, ModuleOp module,
                          unsigned addressSpace) {
  if (func.isDeclaration() || !llvm::hasSingleElement(func.getBody()) ||
      func.getNumArguments() != 3 || func.getNumResults() != 1)
    return func.emitError("tile kernel requires one block, three runtime "
                          "tensor inputs and one result");
  auto tensor = dyn_cast<RankedTensorType>(func.getResultTypes()[0]);
  if (!tensor || !tileTypeForTensor(tensor))
    return func.emitError("requires a static dense FP32 tensor ABI");
  for (Type type : func.getArgumentTypes())
    if (type != tensor)
      return func.emitError(
          "all input/output tensor shapes and types must match");
  for (NamedAttribute attr : func->getAttrs()) {
    auto name = attr.getName().strref();
    if (name != "sym_name" && name != "function_type" &&
        name != "sym_visibility")
      return func.emitError("unsupported function/argument/result attributes "
                            "on tile kernel: ")
             << name;
  }
  auto uses = SymbolTable::getSymbolUses(func, module);
  if (!uses || !uses->empty())
    return func.emitError("tile output-parameter ABI requires an entry "
                          "function with no in-module callers");
  if (func.front().empty() || !isa<func::ReturnOp>(func.front().back()))
    return func.emitError("expected one tensor return");
  auto ret = cast<func::ReturnOp>(func.front().back());
  if (ret.getNumOperands() != 1)
    return ret.emitError("expected one tensor return");
  auto to = ret.getOperand(0).getDefiningOp<aiputile::ToTensorOp>();
  if (!to || !to.getOutput().hasOneUse())
    return ret.emitError("return must use one tensor ABI boundary conversion");
  auto final = to.getInput().getDefiningOp<TileLoopOp>();
  if (!final)
    return to.emitError("run aiputile-tile before lowering");
  llvm::SmallDenseSet<Value> available;
  bool haveLoop = false;
  for (Operation &op : func.front()) {
    if (auto from = dyn_cast<aiputile::FromTensorOp>(op)) {
      auto arg = dyn_cast<BlockArgument>(from.getInput());
      if (!arg || arg.getOwner() != &func.front())
        return from.emitError(
            "tile inputs must come from runtime function arguments");
      available.insert(from.getOutput());
    } else if (auto loop = dyn_cast<TileLoopOp>(op)) {
      if (loop.getOutput().getType().getCapacity() != tensor.getNumElements())
        return loop.emitError(
            "tile loop domain must cover the complete function result");
      for (Value input : loop.getInputs())
        if (!available.contains(input))
          return loop.emitError("tile inputs must be earlier tile loops or "
                                "runtime input boundaries");
      if (addressSpace == 1 && loop != final)
        return loop.emitError("device lowering requires fusion: intermediate "
                              "device allocation is not supported");
      available.insert(loop.getOutput());
      haveLoop = true;
    } else if (auto boundary = dyn_cast<aiputile::ToTensorOp>(op)) {
      if (boundary != to)
        return boundary.emitError(
            "to_tensor is supported only at the returned ABI boundary");
    } else if (!isa<func::ReturnOp>(op)) {
      return op.emitError("only runtime boundaries, tile_loop and return are "
                          "supported in a tile kernel");
    }
  }
  if (!haveLoop)
    return func.emitError("expected at least one partitioned tile_loop");
  return success();
}

SmallVector<Value> globalIndices(OpBuilder &builder, Location loc, Value flat,
                                 MemRefType type) {
  SmallVector<Value> indices(type.getRank());
  Value quotient = flat;
  for (int64_t d = type.getRank() - 1; d >= 0; --d) {
    Value size =
        builder.create<arith::ConstantIndexOp>(loc, type.getDimSize(d));
    indices[d] = builder.create<arith::RemUIOp>(loc, quotient, size);
    if (d)
      quotient = builder.create<arith::DivUIOp>(loc, quotient, size);
  }
  return indices;
}

void lowerKernel(func::FuncOp oldFunc, unsigned addressSpace) {
  auto tensor = cast<RankedTensorType>(oldFunc.getResultTypes()[0]);
  auto ret = cast<func::ReturnOp>(oldFunc.front().back());
  auto to = ret.getOperand(0).getDefiningOp<aiputile::ToTensorOp>();
  Value final = to.getInput();
  OpBuilder builder(oldFunc);
  Location loc = oldFunc.getLoc();
  auto memref = MemRefType::get(tensor.getShape(), tensor.getElementType(),
                                MemRefLayoutAttrInterface{},
                                builder.getI64IntegerAttr(addressSpace));
  auto func = builder.create<func::FuncOp>(
      loc, oldFunc.getSymName(),
      builder.getFunctionType(SmallVector<Type>(4, memref), {}));
  if (auto visibility = oldFunc->getAttr("sym_visibility"))
    func->setAttr("sym_visibility", visibility);
  func.setArgAttr(3, "llvm.noalias", builder.getUnitAttr());
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  DenseMap<Value, Value> buffers;
  SmallVector<Value> temporaries;
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value count =
      builder.create<arith::ConstantIndexOp>(loc, tensor.getNumElements());
  int64_t staging = 0;
  for (Operation &op : oldFunc.front()) {
    if (auto from = dyn_cast<aiputile::FromTensorOp>(op)) {
      buffers[from.getOutput()] = entry->getArgument(
          cast<BlockArgument>(from.getInput()).getArgNumber());
      continue;
    }
    auto loop = dyn_cast<TileLoopOp>(op);
    if (!loop)
      continue;
    Value output;
    if (loop.getOutput() == final)
      output = entry->getArgument(3);
    else {
      output = builder.create<memref::AllocOp>(loc, memref);
      temporaries.push_back(output);
    }
    buffers[loop.getOutput()] = output;
    int64_t size = partitionSize(loop);
    int64_t tiles = 1 + (tensor.getNumElements() - 1) / size;
    int64_t localValues = loop.getInputs().size() +
                          std::distance(loop.getBody().front().begin(),
                                        loop.getBody().front().end()) -
                          1;
    staging = std::max(staging, localValues * size * 4);
    Value step = builder.create<arith::ConstantIndexOp>(loc, size);
    Value tileCount = builder.create<arith::ConstantIndexOp>(loc, tiles);
    builder.create<scf::ForOp>(
        loc, zero, tileCount, one, ValueRange{},
        [&](OpBuilder &outer, Location loc, Value tileId, ValueRange) {
          Value offset = outer.create<arith::MulIOp>(loc, tileId, step);
          Value remaining = outer.create<arith::SubIOp>(loc, count, offset);
          Value valid = outer.create<arith::MinUIOp>(loc, remaining, step);
          outer.create<scf::ForOp>(
              loc, zero, valid, one, ValueRange{},
              [&](OpBuilder &inner, Location loc, Value lane, ValueRange) {
                Value flat = inner.create<arith::AddIOp>(loc, offset, lane);
                DenseMap<Value, Value> scalars;
                auto &body = loop.getBody().front();
                for (auto input : llvm::enumerate(loop.getInputs())) {
                  Value buffer = buffers.lookup(input.value());
                  scalars[body.getArgument(input.index())] =
                      inner.create<memref::LoadOp>(
                          loc, buffer, globalIndices(inner, loc, flat, memref));
                }
                scalars[body.getArgument(loop.getInputs().size())] = offset;
                scalars[body.getArgument(loop.getInputs().size() + 1)] = valid;
                for (Operation &tileOp : body.without_terminator()) {
                  Value lhs = scalars.lookup(tileOp.getOperand(0));
                  Value rhs = scalars.lookup(tileOp.getOperand(1));
                  Value value;
                  if (isa<aiputile::AddOp>(tileOp))
                    value = inner.create<arith::AddFOp>(loc, lhs, rhs);
                  else
                    value = inner.create<arith::MulFOp>(loc, lhs, rhs);
                  scalars[tileOp.getResult(0)] = value;
                }
                Value result = scalars.lookup(
                    cast<aiputile::YieldOp>(body.back()).getValue());
                inner.create<memref::StoreOp>(
                    loc, result, output,
                    globalIndices(inner, loc, flat, memref));
                inner.create<scf::YieldOp>(loc);
              });
          outer.create<scf::YieldOp>(loc);
        });
  }
  // Every use of each unfused temporary completes before it is released.
  for (Value temporary : temporaries)
    builder.create<memref::DeallocOp>(loc, temporary);
  func->setAttr("aiputile.staging_bytes", builder.getI64IntegerAttr(staging));
  builder.create<func::ReturnOp>(loc);
  oldFunc.erase();
}

struct TileLoweringPass
    : PassWrapper<TileLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileLoweringPass)
  TileLoweringPass() = default;
  TileLoweringPass(const TileLoweringPass &other) : PassWrapper(other) {}
  explicit TileLoweringPass(unsigned space) { addressSpace = space; }
  Option<unsigned> addressSpace{
      *this, "device-address-space",
      llvm::cl::desc("ABI memory space: 0 host, 1 Zhouyi device (not SRAM)"),
      llvm::cl::init(0)};
  StringRef getArgument() const final { return "aiputile-lower-tiles"; }
  StringRef getDescription() const final {
    return "Lower symbolic tile regions to scalar SCF loops, materializing "
           "only unfused inter-region results";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    if (addressSpace > 1) {
      getOperation().emitError(
          "supported ABI address spaces are 0 (host) and 1 (Zhouyi device)");
      return signalPassFailure();
    }
    SmallVector<func::FuncOp> funcs;
    for (Operation &op : getOperation().getBody()->getOperations()) {
      auto func = dyn_cast<func::FuncOp>(op);
      if (!func || failed(checkKernel(func, getOperation(), addressSpace))) {
        if (!func)
          op.emitError("tile lowering only accepts entry functions");
        return signalPassFailure();
      }
      funcs.push_back(func);
    }
    if (funcs.empty()) {
      getOperation().emitError("expected at least one tile kernel");
      return signalPassFailure();
    }
    for (auto func : funcs)
      lowerKernel(func, addressSpace);
  }
};

struct TileToLLVMPass : PassWrapper<TileToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileToLLVMPass)
  TileToLLVMPass() = default;
  TileToLLVMPass(const TileToLLVMPass &other) : PassWrapper(other) {}
  TileToLLVMPass(bool bare, unsigned width) {
    barePointers = bare;
    indexBitwidth = width;
  }
  Option<bool> barePointers{*this, "bare-pointers", llvm::cl::init(true)};
  Option<unsigned> indexBitwidth{*this, "index-bitwidth", llvm::cl::init(64)};
  StringRef getArgument() const final { return "aiputile-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower SCF/Arith/MemRef/Func to LLVM with a consistent pointer ABI";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, cf::ControlFlowDialect>();
  }
  void runOnOperation() override {
    if (indexBitwidth != 32 && indexBitwidth != 64) {
      getOperation().emitError("index-bitwidth must be 32 or 64");
      return signalPassFailure();
    }
    LowerToLLVMOptions options(&getContext());
    options.useBarePtrCallConv = barePointers;
    options.overrideIndexBitwidth(indexBitwidth);
    LLVMTypeConverter converter(&getContext(), options);
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    RewritePatternSet patterns(&getContext());
    populateSCFToControlFlowConversionPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(converter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> aiputile::createLowerAipugcToAIPUTilePass() {
  return std::make_unique<ImportPass>();
}
std::unique_ptr<Pass> aiputile::createTileTilingPass(int64_t size) {
  return std::make_unique<TilingPass>(size);
}
std::unique_ptr<Pass> aiputile::createTileFusionPass() {
  return std::make_unique<FusionPass>();
}
std::unique_ptr<Pass> aiputile::createTileLoweringPass(unsigned space) {
  return std::make_unique<TileLoweringPass>(space);
}
std::unique_ptr<Pass> aiputile::createTileToLLVMPass(bool bare,
                                                     unsigned width) {
  return std::make_unique<TileToLLVMPass>(bare, width);
}
void aiputile::registerAIPUTilePasses() {
  PassRegistration<ImportPass>();
  PassRegistration<TilingPass>();
  PassRegistration<FusionPass>();
  PassRegistration<TileLoweringPass>();
  PassRegistration<TileToLLVMPass>();
  PassPipelineRegistration<>(
      "aiputile-fuse-add-mul",
      "Compatibility alias for tile-region fusion; run aiputile-tile first",
      [](OpPassManager &pm) { pm.addPass(createTileFusionPass()); });
}
