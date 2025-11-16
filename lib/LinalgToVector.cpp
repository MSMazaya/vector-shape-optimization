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

    // Loop for j with vector stride (columns processed in chunks of vectorLen)
    auto jLoop = rewriter.create<scf::ForOp>(loc, zero, nConst, vectorLenConst);
    rewriter.setInsertionPointToStart(jLoop.getBody());
    auto jBase = jLoop.getInductionVar();

    // Initialize accumulator vector with zeros (inside j loop)
    auto zeroElement = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getZeroAttr(elementType));
    // Create a broadcast/splat operation to initialize vector with zero
    auto accInit = rewriter.create<vector::BroadcastOp>(loc, vectorType, zeroElement);

    // Inner loop for k dimension (reduction dimension) with accumulator
    SmallVector<Value> initArgs;
    initArgs.push_back(accInit);
    auto innerLoopK = rewriter.create<scf::ForOp>(
        loc, zero, kConst, one, initArgs);
    rewriter.setInsertionPointToStart(innerLoopK.getBody());
    auto k = innerLoopK.getInductionVar();
    // The accumulator is the first iter arg (arg 0 is the induction variable)
    Value accVector = innerLoopK.getBody()->getArgument(1);

    // Load scalar from lhs: A[i, k]
    SmallVector<Value> lhsIndices;
    lhsIndices.push_back(i);
    lhsIndices.push_back(k);
    auto lhsScalar = rewriter.create<memref::LoadOp>(loc, elementType, lhs,
                                                       lhsIndices);
    // Broadcast to vector: replicate A[i,k] across vector
    auto lhsVector = rewriter.create<vector::BroadcastOp>(loc, vectorType, lhsScalar);

    // Load vector from rhs: B[k, j:j+vectorLen-1]
    SmallVector<Value> rhsIndices;
    rhsIndices.push_back(k);
    rhsIndices.push_back(jBase);
    auto rhsVector = rewriter.create<vector::LoadOp>(loc, vectorType, rhs, rhsIndices);

    // Vector multiply-add: acc = acc + lhs * rhs
    auto mulResult = rewriter.create<vector::FMAOp>(loc, vectorType, lhsVector, rhsVector, accVector);
    Value newAcc = mulResult;

    // Yield the updated accumulator
    SmallVector<Value> yieldArgs;
    yieldArgs.push_back(newAcc);
    rewriter.create<scf::YieldOp>(loc, yieldArgs);

    // Get the final accumulator value (after the k loop completes)
    rewriter.setInsertionPointAfter(innerLoopK);
    Value finalAcc = innerLoopK.getResults()[0];

    // Store result vector: C[i, j:j+vectorLen-1] = acc
    SmallVector<Value> resultIndices;
    resultIndices.push_back(i);
    resultIndices.push_back(jBase);
    rewriter.create<vector::StoreOp>(loc, finalAcc, result, resultIndices);

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

