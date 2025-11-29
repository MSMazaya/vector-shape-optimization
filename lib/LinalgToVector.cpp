#include "my/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::vector;

namespace {

// Unrolling option
static llvm::cl::opt<bool> unrollScalarKOpt(
    "linalg-to-vector-unroll-scalar-k",
    llvm::cl::desc("Manually unroll scalar remainder inner-k loop"),
    llvm::cl::init(false));

// Remainder Masking option
static llvm::cl::opt<bool> useMaskedRemainderOpt(
    "linalg-to-vector-use-masked-remainder",
    llvm::cl::desc("Use masked vector operations for remainder handling"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> useFullyMaskedOpt(
  "linalg-to-vector-fully-masked",
  llvm::cl::desc("Use fully masked method"),
  llvm::cl::init(false)
);

// Vector width option (128 for AVX, 512 for AVX-512)
static llvm::cl::opt<int> vectorWidthOpt(
    "linalg-to-vector-vector-width",
    llvm::cl::desc("Vector width in bits (128 for AVX, 512 for AVX-512)"),
    llvm::cl::init(128));

// Stride option for fully masked mode
static llvm::cl::opt<int> maskedStrideOpt(
    "linalg-to-vector-masked-stride",
    llvm::cl::desc("Stride for fully masked mode (in elements). If specified and leaves no remainder, use it; otherwise use max hardware size."),
    llvm::cl::init(0));

// Cache line size option for fully masked mode
static llvm::cl::opt<bool> useCacheLineStrideOpt(
    "linalg-to-vector-use-cache-line-stride",
    llvm::cl::desc("Use cache line size (64 bytes) as stride heuristic for fully masked mode"),
    llvm::cl::init(false));

// Calculate the vector length based on element type and vector width
int64_t getVectorLength(Type elementType, int vectorWidthBits) {
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  return vectorWidthBits / bitWidth;
}

// Calculate cache line size stride (64 bytes / element size), capped at vectorLen
int64_t getCacheLineStride(Type elementType, int64_t vectorLen) {
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  unsigned byteWidth = bitWidth / 8;
  int64_t cacheLineBytes = 64;
  int64_t cacheLineStride = cacheLineBytes / byteWidth;
  // Cap at hardware vector width
  return std::min(cacheLineStride, vectorLen);
}

// Pattern to convert MatmulOp to vector operations
struct MatmulToVectorPattern : public OpRewritePattern<linalg::MatmulOp> {
  MatmulToVectorPattern(MLIRContext *context, bool unrollScalarK, bool useMaskedRemainder, bool useFullyMasked)
      : OpRewritePattern<linalg::MatmulOp>(context),
        unrollScalarK(unrollScalarK),
        useMaskedRemainder(useMaskedRemainder),
        useFullyMasked(useFullyMasked) {}

  bool unrollScalarK;
  bool useMaskedRemainder;
  bool useFullyMasked;

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
    int64_t vectorLen = getVectorLength(elementType, vectorWidthOpt);

    // Create vector type
    VectorType vectorType = VectorType::get({vectorLen}, elementType);

    // Get dimension sizes
    int64_t m = resultType.getDimSize(0);
    int64_t n = resultType.getDimSize(1);
    int64_t kDim = lhsType.getDimSize(1);

    // Check if dimensions are static and if there's a remainder at compile time
    bool hasStaticDims =
        !resultType.isDynamicDim(0) && !resultType.isDynamicDim(1);
    bool hasRemainderAtCompileTime = hasStaticDims && (n % vectorLen != 0);

    // Create constants
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto vectorLenConst =
        rewriter.create<arith::ConstantIndexOp>(loc, vectorLen);
    auto mConst = rewriter.create<arith::ConstantIndexOp>(loc, m);
    auto nConst = rewriter.create<arith::ConstantIndexOp>(loc, n);
    auto kConst = rewriter.create<arith::ConstantIndexOp>(loc, kDim);

    auto buildMaskedRemainder = [&](Value iIdx, Value jStart, Value jEnd) {
      // calculate number of remaining elements
      Value remainderSize = rewriter.create<arith::SubIOp>(loc, jEnd, jStart);

      // create mask based on remainder size
      auto maskType = VectorType::get({vectorLen}, rewriter.getI1Type());
      Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, remainderSize);

      // Initialize accumulator with zeros
      auto zeroElement = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getZeroAttr(elementType)
      );
      auto zeroVec = rewriter.create<vector::BroadcastOp>(loc, vectorType, zeroElement);

      // k-loop with masked operatioins
      SmallVector<Value> initArgs{zeroVec};
      auto kLoop = rewriter.create<scf::ForOp>(loc, zero, kConst, one, initArgs);
      
      // TODO: comment needed
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value k = kLoop.getInductionVar();
      Value accVec = kLoop.getBody()->getArgument(1);

      // Load A[i,k] and broadcast
      SmallVector<Value> lhsIdx{iIdx, k};
      auto aScalar = rewriter.create<memref::LoadOp>(loc, elementType, lhs, lhsIdx);
      auto aVec = rewriter.create<vector::BroadcastOp>(loc, vectorType, aScalar);

      // Masked load B[k, jStart:jStart+vectorLen]
      SmallVector<Value> rhsIdx{k, jStart};
      auto bVec = rewriter.create<vector::MaskedLoadOp>(
        loc, vectorType, rhs, rhsIdx, mask, zeroVec
      );

      // FMA: acc = acc + a*b
      auto fmaResult = rewriter.create<vector::FMAOp>(loc, vectorType, aVec, bVec, accVec);

      rewriter.create<scf::YieldOp>(loc, ValueRange{fmaResult.getResult()});

      // Store result with mask
      rewriter.setInsertionPointAfter(kLoop);
      Value finalVec = kLoop.getResults()[0];

      SmallVector<Value> resultIdx{iIdx, jStart};
      rewriter.create<vector::MaskedStoreOp>(loc, result, resultIdx, mask, finalVec);
    };
    
    // TODO: remove this into a function? too much hassle though
    // Helper: scalar k-loop (with optional unrolling)
    auto buildScalarK = [&](Value iIdx, Value jIdx) -> Value {
      auto zeroVal = rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getZeroAttr(elementType));

      // Helper to do one k iteration
      auto doOneIteration = [&](Value kIdx, Value accVal) -> Value {
        SmallVector<Value> lhsIdx{iIdx, kIdx};
        auto aVal =
            rewriter.create<memref::LoadOp>(loc, elementType, lhs, lhsIdx);

        SmallVector<Value> rhsIdx{kIdx, jIdx};
        auto bVal =
            rewriter.create<memref::LoadOp>(loc, elementType, rhs, rhsIdx);

        auto mul = rewriter.create<arith::MulFOp>(loc, aVal, bVal);
        return rewriter.create<arith::AddFOp>(loc, accVal, mul);
      };

      // Check if we can fully unroll (static K dimension and unroll enabled)
      bool canFullyUnroll =
          unrollScalarK && !lhsType.isDynamicDim(1) && kDim > 0;

      if (canFullyUnroll) {
        // Fully unrolling: generate explicit iterations for each k value
        Value acc = zeroVal;
        for (int64_t k = 0; k < kDim; k++) {
          auto kConst = rewriter.create<arith::ConstantIndexOp>(loc, k);
          acc = doOneIteration(kConst, acc);
        }
        return acc;
      } else {
        // Use a scalar loop remainder
        SmallVector<Value> initArgs{zeroVal};
        Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto kLoop =
            rewriter.create<scf::ForOp>(loc, zero, kConst, step, initArgs);

        rewriter.setInsertionPointToStart(kLoop.getBody());
        Value kBase = kLoop.getInductionVar();
        Value acc = kLoop.getBody()->getArgument(1);

        acc = doOneIteration(kBase, acc);

        rewriter.create<scf::YieldOp>(loc, acc);
        rewriter.setInsertionPointAfter(kLoop);

        return kLoop.getResults()[0];
      }
    };

    // Main vectorized loops:
    if (useFullyMasked) {
      // Calculate effective stride based on heuristics:
      // 1. If stride is specified and leaves no remainder, use it (capped at hardware max)
      // 2. If cache line stride is enabled and leaves no remainder, use it (capped at hardware max)
      // 3. Otherwise, use max hardware size (vectorLen)
      int64_t effectiveStride = vectorLen;
      bool useCustomStride = false;
      
      // Calculate cache line stride if enabled
      int64_t cacheLineStride = 0;
      if (useCacheLineStrideOpt) {
        cacheLineStride = getCacheLineStride(elementType, vectorLen);
      }
      
      if (maskedStrideOpt > 0) {
        if (hasStaticDims) {
          // Compile-time check: if stride leaves no remainder and is within hardware limits
          if (n % maskedStrideOpt == 0 && maskedStrideOpt <= vectorLen) {
            effectiveStride = maskedStrideOpt;
            useCustomStride = true;
          }
        } else {
          // For dynamic dimensions, we'll check at runtime
          // For now, we'll use the specified stride if it's within hardware limits
          if (maskedStrideOpt <= vectorLen) {
            effectiveStride = maskedStrideOpt;
            useCustomStride = true;
          }
        }
      } else if (useCacheLineStrideOpt && cacheLineStride > 0) {
        // Try cache line stride if no explicit stride specified
        if (hasStaticDims) {
          // Compile-time check: if cache line stride leaves no remainder
          if (n % cacheLineStride == 0 && cacheLineStride <= vectorLen) {
            effectiveStride = cacheLineStride;
            useCustomStride = true;
          }
        } else {
          // For dynamic dimensions, we'll check at runtime
          if (cacheLineStride <= vectorLen) {
            effectiveStride = cacheLineStride;
            useCustomStride = true;
          }
        }
      }
      
      // Create vector type and stride constant
      VectorType effectiveVectorType;
      Value finalStride;
      
      if (hasStaticDims && useCustomStride) {
        // Static dimensions: use effective stride as vector width
        effectiveVectorType = VectorType::get({effectiveStride}, elementType);
        finalStride = rewriter.create<arith::ConstantIndexOp>(loc, effectiveStride);
      } else {
        // Dynamic dimensions or no custom stride: use hardware max
        effectiveVectorType = vectorType;
        finalStride = vectorLenConst;
        
        // For dynamic dimensions with custom stride, add runtime check
        if (!hasStaticDims) {
          Value strideCandidate = vectorLenConst;
          
          if (maskedStrideOpt > 0 && maskedStrideOpt <= vectorLen) {
            strideCandidate = rewriter.create<arith::ConstantIndexOp>(loc, maskedStrideOpt);
          } else if (useCacheLineStrideOpt && cacheLineStride > 0 && cacheLineStride <= vectorLen) {
            strideCandidate = rewriter.create<arith::ConstantIndexOp>(loc, cacheLineStride);
          }
          
          if (strideCandidate != vectorLenConst) {
            auto strideRemainder = rewriter.create<arith::RemUIOp>(loc, nConst, strideCandidate);
            auto noRemainder = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, strideRemainder, zero);
            
            // Select stride: if no remainder, use candidate stride; otherwise use vectorLen
            finalStride = rewriter.create<arith::SelectOp>(
                loc, noRemainder, strideCandidate, vectorLenConst);
          }
        }
      }
      
      // Outer loop over rows
      auto outerLoopI = rewriter.create<scf::ForOp>(loc, zero, mConst, one);
      rewriter.setInsertionPointToStart(outerLoopI.getBody());
      Value i = outerLoopI.getInductionVar();
      
      // Single J-loop with masking for every iteration
      // Use final stride for loop increment
      auto jLoop = rewriter.create<scf::ForOp>(loc, zero, nConst, finalStride);
      rewriter.setInsertionPointToStart(jLoop.getBody());
      Value j = jLoop.getInductionVar();
      
      // Calculate active elements: min(finalStride, n - j)
      Value remainingElements = rewriter.create<arith::SubIOp>(loc, nConst, j);
      Value activeCount = rewriter.create<arith::MinSIOp>(loc, finalStride, remainingElements);
      
      // Create mask - use effective vector type size
      // For static dimensions with custom stride, use effectiveStride
      // For dynamic or default case, use vectorLen (hardware max)
      auto maskType = hasStaticDims && useCustomStride 
          ? VectorType::get({effectiveStride}, rewriter.getI1Type())
          : VectorType::get({vectorLen}, rewriter.getI1Type());
      Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, activeCount);
      
      // Zero vector with effective vector type
      auto zeroElement = rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getZeroAttr(elementType));
      auto zeroVec = rewriter.create<vector::BroadcastOp>(loc, effectiveVectorType, zeroElement);
      
      // K-loop
      SmallVector<Value> initArgs{zeroVec};
      auto kLoop = rewriter.create<scf::ForOp>(loc, zero, kConst, one, initArgs);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value k = kLoop.getInductionVar();
      Value accVec = kLoop.getBody()->getArgument(1);
      
      // Load and broadcast A[i,k]
      SmallVector<Value> lhsIdx{i, k};
      auto aScalar = rewriter.create<memref::LoadOp>(loc, elementType, lhs, lhsIdx);
      auto aVec = rewriter.create<vector::BroadcastOp>(loc, effectiveVectorType, aScalar);
      
      // Masked load B[k, j:j+finalStride]
      SmallVector<Value> rhsIdx{k, j};
      auto bVec = rewriter.create<vector::MaskedLoadOp>(
          loc, effectiveVectorType, rhs, rhsIdx, mask, zeroVec);
      
      // FMA
      auto fmaResult = rewriter.create<vector::FMAOp>(loc, effectiveVectorType, aVec, bVec, accVec);
      rewriter.create<scf::YieldOp>(loc, ValueRange{fmaResult.getResult()});
      
      // Masked store
      rewriter.setInsertionPointAfter(kLoop);
      Value finalVec = kLoop.getResults()[0];
      SmallVector<Value> resultIdx{i, j};
      rewriter.create<vector::MaskedStoreOp>(loc, result, resultIdx, mask, finalVec);
      
      // jLoop doesn't need a yield since it has no iter_args
      // Just move insertion point after it
      rewriter.setInsertionPointAfter(jLoop);
      
      // outerLoopI also doesn't need a yield since it has no iter_args
      rewriter.setInsertionPointAfter(outerLoopI);
      rewriter.eraseOp(matmulOp);
      return success();
    }


    // Create loops for vectorized matmul
    // Outer loop for i (rows)
    auto outerLoopI = rewriter.create<scf::ForOp>(loc, zero, mConst, one);
    rewriter.setInsertionPointToStart(outerLoopI.getBody());
    auto i = outerLoopI.getInductionVar();

    // Calculate how many full vectors we can process
    // Process full vectors first, then handle remainder separately
    auto nDivVectorLen =
        rewriter.create<arith::DivUIOp>(loc, nConst, vectorLenConst);
    auto fullVectorCount =
        rewriter.create<arith::MulIOp>(loc, nDivVectorLen, vectorLenConst);

    // Check if we can vectorize at all (n >= vectorLen)
    // For static dimensions, check at compile time; for dynamic, check at
    // runtime
    bool canVectorize = hasStaticDims ? (n >= vectorLen) : true;

    // Loop for j with vector stride (columns processed in chunks of vectorLen)
    // Only iterate over full vectors to avoid out-of-bounds access
    // Only generate vectorized loop if n >= vectorLen
    scf::ForOp jLoop;
    if (canVectorize || !hasStaticDims) {
      // For dynamic dimensions, we need a runtime check
      if (!hasStaticDims) {
        auto canVectorizeCond = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::uge, nConst, vectorLenConst);
        auto vectorizeIf =
            rewriter.create<scf::IfOp>(loc, canVectorizeCond, false);
        rewriter.setInsertionPointToStart(vectorizeIf.thenBlock());
      }

      jLoop = rewriter.create<scf::ForOp>(loc, zero, fullVectorCount,
                                          vectorLenConst);
      rewriter.setInsertionPointToStart(jLoop.getBody());
      auto jBase = jLoop.getInductionVar();

      // Initialize accumulator vector with zeros
      auto zeroElement = rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getZeroAttr(elementType));
      auto accInit =
          rewriter.create<vector::BroadcastOp>(loc, vectorType, zeroElement);

      // Inner loop for k dimension (reduction dimension) with accumulator
      SmallVector<Value> initArgs;
      initArgs.push_back(accInit);
      auto innerLoopK =
          rewriter.create<scf::ForOp>(loc, zero, kConst, one, initArgs);
      rewriter.setInsertionPointToStart(innerLoopK.getBody());
      auto k = innerLoopK.getInductionVar();
      Value accVector = innerLoopK.getBody()->getArgument(1);

      // Load scalar from lhs: A[i, k]
      SmallVector<Value> lhsIndices;
      lhsIndices.push_back(i);
      lhsIndices.push_back(k);
      auto lhsScalar =
          rewriter.create<memref::LoadOp>(loc, elementType, lhs, lhsIndices);

      // Broadcast to vector: replicate A[i,k] across vector
      auto lhsVector =
          rewriter.create<vector::BroadcastOp>(loc, vectorType, lhsScalar);

      // Load vector from rhs: B[k, j:j+vectorLen-1]
      SmallVector<Value> rhsIndices;
      rhsIndices.push_back(k);
      rhsIndices.push_back(jBase);
      auto rhsVector =
          rewriter.create<vector::LoadOp>(loc, vectorType, rhs, rhsIndices);

      // Vector multiply-add: acc = acc + lhs * rhs
      auto mulResult = rewriter.create<vector::FMAOp>(
          loc, vectorType, lhsVector, rhsVector, accVector);

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
    }

    // Remainder:
    if (canVectorize || !hasStaticDims) {
      rewriter.setInsertionPointAfter(jLoop);
    } else {
      rewriter.setInsertionPointToStart(outerLoopI.getBody());
    }

    if (hasStaticDims && !hasRemainderAtCompileTime)
      // Don't want to forget this edge case here, thus the goto...
      goto erase_and_done;

    // For dynamic dimensions, we need a runtime check
    if (!hasStaticDims) {
      // Runtime check if fullVectorCount < n
      auto hasRemainderCond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, fullVectorCount, nConst);
      auto remainderIf =
          rewriter.create<scf::IfOp>(loc, hasRemainderCond, false);
      rewriter.setInsertionPointToStart(remainderIf.thenBlock());
    }

    // // Remainder loop
    // {
    //   auto remLoop =
    //       rewriter.create<scf::ForOp>(loc, fullVectorCount, nConst, one);
    //   rewriter.setInsertionPointToStart(remLoop.getBody());
    //   Value jRem = remLoop.getInductionVar();

    //   Value finalScalar = buildScalarK(i, jRem);

    //   rewriter.create<memref::StoreOp>(loc, finalScalar, result,
    //                                    ValueRange{i, jRem});
    // }
    // remainder:
    if (canVectorize || !hasStaticDims) {
      rewriter.setInsertionPointAfter(jLoop);
    } else {
      rewriter.setInsertionPointToStart(outerLoopI.getBody());
    }
    if (hasStaticDims && !hasRemainderAtCompileTime) goto erase_and_done;

    // dynamic Dimensions, runtime check
    if (!hasStaticDims) {
      auto hasRemainderCond = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, fullVectorCount, nConst
      );
      auto remainderIf = rewriter.create<scf::IfOp>(loc, hasRemainderCond, nConst);
      rewriter.setInsertionPointToStart(remainderIf.thenBlock());
    }
    
    // choose masked or scalar remainder
    if (useMaskedRemainder) {
      buildMaskedRemainder(i, fullVectorCount, nConst);
    } else {
      auto remLoop = rewriter.create<scf::ForOp>(loc, fullVectorCount, nConst, one);
      rewriter.setInsertionPointToStart(remLoop.getBody());
      Value jRem = remLoop.getInductionVar();
      Value finalScalar = buildScalarK(i, jRem);
      rewriter.create<memref::StoreOp>(loc, finalScalar, result, ValueRange{i, jRem});
    }
  erase_and_done:

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
    return "Convert linalg operations to vector instructions with configurable "
           "vector width (default 128-bit for AVX, use --linalg-to-vector-vector-width=512 for AVX-512)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, func::FuncDialect, vector::VectorDialect,
                arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    bool shouldUnroll =
        unrollScalarKOpt || (getenv("UNROLL_REMAINDER") != nullptr);
    bool useMaskedRemainder = useMaskedRemainderOpt ||
        (getenv("USE_MASKED_REMAINDER") != nullptr);
    bool useFullyMasked = useFullyMaskedOpt;
        
    patterns.add<MatmulToVectorPattern>(context, shouldUnroll, useMaskedRemainder, useFullyMasked);

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
