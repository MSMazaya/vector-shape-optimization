#include "my/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::vector;

namespace {

// Calculate the vector length for 128-bit vectors based on element type
int64_t getVectorLength(Type elementType) {
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  // For 128-bit vectors: 128 / bit_width
  return 128 / bitWidth;
}

// Pattern to convert MatmulOp to vector operations
struct MatmulToVectorPattern : public OpRewritePattern<linalg::MatmulOp> {
  MatmulToVectorPattern(MLIRContext *context)
      : OpRewritePattern<linalg::MatmulOp>(context) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();

    // Get operands
    Value lhs = matmulOp.getInputs()[0];
    Value rhs = matmulOp.getInputs()[1];
    Value result = matmulOp.getOutputs()[0];

    auto lhsType = dyn_cast<MemRefType>(lhs.getType());
    auto rhsType = dyn_cast<MemRefType>(rhs.getType());
    auto resultType = dyn_cast<MemRefType>(result.getType());
    
    if (!lhsType || !rhsType || !resultType) {
      return failure();
    }

    Type elementType = lhsType.getElementType();
    int64_t vectorLen = getVectorLength(elementType);

    // Create vector type for 128-bit vectors
    VectorType vectorType = VectorType::get({vectorLen}, elementType);

    // Get dimension sizes
    int64_t m = resultType.getDimSize(0);
    int64_t n = resultType.getDimSize(1);
    int64_t kDim = lhsType.getDimSize(1);
    
    // Check if dimensions are static and if there's a remainder at compile time
    bool hasStaticDims = !resultType.isDynamicDim(0) && !resultType.isDynamicDim(1);
    bool hasRemainderAtCompileTime = hasStaticDims && (n % vectorLen != 0);

    // Create constants
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto vectorLenConst = rewriter.create<arith::ConstantIndexOp>(loc, vectorLen);
    auto mConst = rewriter.create<arith::ConstantIndexOp>(loc, m);
    auto nConst = rewriter.create<arith::ConstantIndexOp>(loc, n);
    auto kConst = rewriter.create<arith::ConstantIndexOp>(loc, kDim);

    // Create loops for vectorized matmul
    // Outer loop for i (rows)
    auto outerLoopI = rewriter.create<scf::ForOp>(loc, zero, mConst, one);
    rewriter.setInsertionPointToStart(outerLoopI.getBody());
    auto i = outerLoopI.getInductionVar();

    // Calculate how many full vectors we can process
    // Process full vectors first, then handle remainder separately
    auto nDivVectorLen = rewriter.create<arith::DivUIOp>(loc, nConst, vectorLenConst);
    auto fullVectorCount = rewriter.create<arith::MulIOp>(loc, nDivVectorLen, vectorLenConst);
    
    // Loop for j with vector stride (columns processed in chunks of vectorLen)
    // Only iterate over full vectors to avoid out-of-bounds access
    auto jLoop = rewriter.create<scf::ForOp>(loc, zero, fullVectorCount, vectorLenConst);
    rewriter.setInsertionPointToStart(jLoop.getBody());
    auto jBase = jLoop.getInductionVar();

    // Initialize accumulator vector with zeros
    auto zeroElement = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getZeroAttr(elementType));
    auto accInit = rewriter.create<vector::BroadcastOp>(loc, vectorType, zeroElement);

    // Inner loop for k dimension (reduction dimension) with accumulator
    SmallVector<Value> initArgs;
    initArgs.push_back(accInit);
    auto innerLoopK = rewriter.create<scf::ForOp>(
        loc, zero, kConst, one, initArgs);
    rewriter.setInsertionPointToStart(innerLoopK.getBody());
    auto k = innerLoopK.getInductionVar();
    Value accVector = innerLoopK.getBody()->getArgument(1);

    // Load scalar from lhs: A[i, k]
    SmallVector<Value> lhsIndices;
    lhsIndices.push_back(i);
    lhsIndices.push_back(k);
    auto lhsScalar = rewriter.create<memref::LoadOp>(loc, elementType, lhs, lhsIndices);
    
    // Broadcast to vector: replicate A[i,k] across vector
    auto lhsVector = rewriter.create<vector::BroadcastOp>(loc, vectorType, lhsScalar);

    // Load vector from rhs: B[k, j:j+vectorLen-1]
    SmallVector<Value> rhsIndices;
    rhsIndices.push_back(k);
    rhsIndices.push_back(jBase);
    auto rhsVector = rewriter.create<vector::LoadOp>(loc, vectorType, rhs, rhsIndices);

    // Vector multiply-add: acc = acc + lhs * rhs
    auto mulResult = rewriter.create<vector::FMAOp>(loc, vectorType, lhsVector, rhsVector, accVector);
    
    // Yield the updated accumulator
    SmallVector<Value> yieldArgs;
    yieldArgs.push_back(mulResult);
    rewriter.create<scf::YieldOp>(loc, yieldArgs);

    // Get the final accumulator value (after the k loop completes)
    rewriter.setInsertionPointAfter(innerLoopK);
    Value finalAcc = innerLoopK.getResults()[0];
    // Store result vector: C[i, j:j+vectorLen-1] = acc
    SmallVector<Value> resultIndices;
    resultIndices.push_back(i);
    resultIndices.push_back(jBase);
    rewriter.create<vector::StoreOp>(loc, finalAcc, result, resultIndices);
    
    // Handle remainder: process remaining columns with scalar operations
    // Only generate remainder loop if there's actually a remainder
    rewriter.setInsertionPointAfter(jLoop);
    
    // For static dimensions with no remainder, skip generating remainder code entirely
    // For static dimensions with remainder, generate remainder loop directly (no if needed)
    // For dynamic dimensions, generate remainder loop with runtime check
    if (hasRemainderAtCompileTime || !hasStaticDims) {
      // For dynamic dimensions, we need a runtime check
      if (!hasStaticDims) {
        // Runtime check: fullVectorCount < n
        auto hasRemainderCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, 
                                                              fullVectorCount, nConst);
        // Conditionally generate remainder loop only if needed
        auto remainderIf = rewriter.create<scf::IfOp>(loc, hasRemainderCond, false);
        rewriter.setInsertionPointToStart(remainderIf.thenBlock());
      }
      
      // Create a scalar loop for the remainder columns (from fullVectorCount to n)
      auto remainderLoop = rewriter.create<scf::ForOp>(loc, fullVectorCount, nConst, one);
      rewriter.setInsertionPointToStart(remainderLoop.getBody());
      auto jRem = remainderLoop.getInductionVar();
      
      // Scalar k-loop for remainder columns
      auto zeroScalar = rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getZeroAttr(elementType));
      SmallVector<Value> scalarInitArgs;
      scalarInitArgs.push_back(zeroScalar);
      auto scalarKLoop = rewriter.create<scf::ForOp>(
          loc, zero, kConst, one, scalarInitArgs);
      rewriter.setInsertionPointToStart(scalarKLoop.getBody());
      auto kScalar = scalarKLoop.getInductionVar();
      Value accScalar = scalarKLoop.getBody()->getArgument(1);
      
      // Load A[i, k]
      SmallVector<Value> lhsScalarIndices;
      lhsScalarIndices.push_back(i);
      lhsScalarIndices.push_back(kScalar);
      auto aVal = rewriter.create<memref::LoadOp>(loc, elementType, lhs, lhsScalarIndices);
      
      // Load B[k, jRem]
      SmallVector<Value> rhsScalarIndices;
      rhsScalarIndices.push_back(kScalar);
      rhsScalarIndices.push_back(jRem);
      auto bVal = rewriter.create<memref::LoadOp>(loc, elementType, rhs, rhsScalarIndices);
      
      // Multiply and add: acc = acc + A[i,k] * B[k,jRem]
      auto mulScalar = rewriter.create<arith::MulFOp>(loc, aVal, bVal);
      auto addScalar = rewriter.create<arith::AddFOp>(loc, accScalar, mulScalar);
      
      // Yield updated accumulator
      SmallVector<Value> scalarYieldArgs;
      scalarYieldArgs.push_back(addScalar);
      rewriter.create<scf::YieldOp>(loc, scalarYieldArgs);
      
      // Get final scalar accumulator
      rewriter.setInsertionPointAfter(scalarKLoop);
      Value finalScalarAcc = scalarKLoop.getResults()[0];
      
      // Store result: C[i, jRem] = acc
      SmallVector<Value> resultScalarIndices;
      resultScalarIndices.push_back(i);
      resultScalarIndices.push_back(jRem);
      rewriter.create<memref::StoreOp>(loc, finalScalarAcc, result, resultScalarIndices);
    }

    // Replace the original matmul operation
    rewriter.eraseOp(matmulOp);

    return success();
  }
};

struct LinalgToVectorPass
    : public PassWrapper<LinalgToVectorPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToVectorPass)

  StringRef getArgument() const final { return "linalg-to-vector"; }

  StringRef getDescription() const final {
    return "Convert linalg operations to vector instructions with 128-bit "
           "vectors";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect,
                    vector::VectorDialect, arith::ArithDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    
    // Add patterns to convert linalg operations to vector operations
    patterns.add<MatmulToVectorPattern>(context);

    // Apply patterns
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createLinalgToVectorPass() {
  return std::make_unique<LinalgToVectorPass>();
}
} // namespace mlir

static PassRegistration<LinalgToVectorPass> LinalgToVectorPassReg;

