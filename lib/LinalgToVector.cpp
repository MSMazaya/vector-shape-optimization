#include "my/Passes.h"
#include "my/SpeedupModel.h"

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

// Debug flag for strategy selection
static llvm::cl::opt<bool> debugStrategyOpt(
    "linalg-to-vector-debug-strategy",
    llvm::cl::desc("Print debug information about strategy selection"),
    llvm::cl::init(false));

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

// Body masking stride option (LS: Logical Size)
// When LS < VS, body is masked with stride LS. Remainder handling is independent.
static llvm::cl::opt<int> bodyMaskingStrideOpt(
    "linalg-to-vector-body-masking-stride",
    llvm::cl::desc("Body masking stride (LS) in elements. When LS < VS, body is masked with stride LS. Remainder handling is independent (use --linalg-to-vector-use-masked-remainder or --linalg-to-vector-unroll-scalar-k)."),
    llvm::cl::init(0));

// Vector width option (128 for AVX, 512 for AVX-512)
static llvm::cl::opt<int> vectorWidthOpt(
    "linalg-to-vector-vector-width",
    llvm::cl::desc("Vector width in bits (128 for AVX, 512 for AVX-512)"),
    llvm::cl::init(128));

// Cache line size option for body masking stride heuristic
static llvm::cl::opt<bool> useCacheLineStrideOpt(
    "linalg-to-vector-use-cache-line-stride",
    llvm::cl::desc("Use cache line size (64 bytes) as stride heuristic for body masking"),
    llvm::cl::init(false));

// Heuristic-based strategy selection
static llvm::cl::opt<bool> useHeuristicStrategyOpt(
    "linalg-to-vector-use-heuristic",
    llvm::cl::desc("Use heuristic-based strategy selection (g(f(input))) instead of explicit flags"),
    llvm::cl::init(false));

// Model-based strategy selection using neural network
static llvm::cl::opt<bool> useModelStrategyOpt(
    "linalg-to-vector-use-model",
    llvm::cl::desc("Use neural network model for strategy selection"),
    llvm::cl::init(false));

// Use MLIR's generic linalg vectorizer (linalg::vectorize) instead of the
// custom MatmulToVectorPattern. This provides an MLIR-baseline implementation
// for linalg.matmul when enabled.
static llvm::cl::opt<bool> useMlirVectorizerOpt(
    "linalg-to-vector-use-mlir-vectorize",
    llvm::cl::desc("Use MLIR linalg::vectorize helper for linalg.matmul"),
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

// Strategy decision structure
struct StrategyDecision {
  enum StrategyType {
    MASK_BODY,      // Fully masked body with strides
    MASK_REMAINDER, // Masked remainder only
    UNROLL_REMAINDER, // Unrolled remainder
    NO_MASKING      // No masking - use regular vector ops (for perfect tiles)
  };
  
  StrategyType strategy;
  int64_t stride;           // Optimal stride for masking/unrolling
  bool useStride;           // Whether to use stride optimization
  bool cacheAligned;        // Whether stride is cache-aligned
  bool eliminatesRemainder; // Whether stride eliminates remainder
  
  StrategyDecision() : strategy(MASK_REMAINDER), stride(0), useStride(false),
                       cacheAligned(false), eliminatesRemainder(false) {}
};

// Input weights for cost function
// Tuned based on benchmark results: masking capability is the most important factor
struct InputWeights {
  double alignmentWeight;      // Weight for alignment/vector size
  double maskingCapabilityWeight; // Weight for ISA masking capability (increased - most important)
  double cacheLineWeight;      // Weight for cache line size
  
  InputWeights() : alignmentWeight(0.2), maskingCapabilityWeight(0.7), cacheLineWeight(0.1) {}
};

// Cost function f(input) - evaluates the cost/benefit of different strategies
struct CostEvaluation {
  double alignmentScore;      // Score based on alignment/vector size
  double maskingScore;        // Score based on masking capability
  double cacheLineScore;      // Score based on cache line alignment
  double remainderCost;       // Cost of handling remainder
  
  CostEvaluation() : alignmentScore(0.0), maskingScore(0.0), 
                     cacheLineScore(0.0), remainderCost(0.0) {}
  
  double weightedCost(const InputWeights& weights) const {
    return (alignmentScore * weights.alignmentWeight +
            maskingScore * weights.maskingCapabilityWeight +
            cacheLineScore * weights.cacheLineWeight) - remainderCost;
  }
};

// Cost function f(input)
CostEvaluation evaluateCost(
    int64_t n, int64_t vectorLen, Type elementType,
    bool hasMaskingCapability, bool hasStaticDims) {
  CostEvaluation cost;
  
  // Alignment score: how well does n align with vectorLen?
  int64_t remainder = n % vectorLen;
  double alignmentRatio = remainder == 0 ? 1.0 : 1.0 - (double)remainder / vectorLen;
  cost.alignmentScore = alignmentRatio;
  
  // Masking score: how beneficial is masking? 
  // Based on results: masking is almost always better when available, especially for AVX-512
  if (hasMaskingCapability) {
    if (remainder == 0) {
      cost.maskingScore = 1.0; // Perfect case
    } else {
      // Masking is highly beneficial - give it a high score
      // Small penalty for very large remainders, but still prefer masking
      double remainderRatio = (double)remainder / vectorLen;
      // More aggressive: masking is almost always better, minimal penalty
      cost.maskingScore = 1.0 - remainderRatio * 0.1; // Very small penalty even for large remainders
    }
  } else {
    cost.maskingScore = 0.0;
  }
  
  // Cache line score: how well does stride align with cache lines?
  int64_t cacheLineStride = getCacheLineStride(elementType, vectorLen);
  if (n % cacheLineStride == 0 && cacheLineStride <= vectorLen) {
    cost.cacheLineScore = 1.0;
  } else {
    // Check if any divisor of n that's <= vectorLen aligns with cache line
    bool foundCacheAligned = false;
    for (int64_t stride = vectorLen; stride > 0; stride--) {
      if (n % stride == 0 && stride == cacheLineStride) {
        foundCacheAligned = true;
        break;
      }
    }
    cost.cacheLineScore = foundCacheAligned ? 0.8 : 0.0;
  }
  
  // Remainder cost: cost of handling remainder (higher for larger remainders)
  if (remainder == 0) {
    cost.remainderCost = 0.0;
  } else {
    double remainderRatio = (double)remainder / vectorLen;
    cost.remainderCost = remainderRatio * 0.3; // Cost increases with remainder size
  }
  
  return cost;
}

// Find optimal stride for masking
int64_t findOptimalMaskingStride(int64_t n, int64_t vectorLen, Type elementType,
                                  bool& cacheAligned, bool& eliminatesRemainder) {
  cacheAligned = false;
  eliminatesRemainder = false;
  
  int64_t cacheLineStride = getCacheLineStride(elementType, vectorLen);
  
  // Priority 1: Cache-aligned stride that eliminates remainder
  if (cacheLineStride > 0 && n % cacheLineStride == 0 && cacheLineStride <= vectorLen) {
    cacheAligned = true;
    eliminatesRemainder = true;
    return cacheLineStride;
  }
  
  // Priority 2: Any stride that eliminates remainder (prefer larger)
  for (int64_t stride = vectorLen; stride > 0; stride--) {
    if (n % stride == 0) {
      eliminatesRemainder = true;
      if (stride == cacheLineStride) {
        cacheAligned = true;
      }
      return stride;
    }
  }
  
  // Priority 3: Cache-aligned stride even if it doesn't eliminate remainder
  if (cacheLineStride > 0 && cacheLineStride <= vectorLen) {
    cacheAligned = true;
    return cacheLineStride;
  }
  
  // Priority 4: Max hardware size
  return vectorLen;
}

// Find optimal stride for unrolling
int64_t findOptimalUnrollingStride(int64_t remainder, int64_t vectorLen) {
  // For unrolling, we want to minimize the number of iterations
  // Prefer strides that divide the remainder evenly
  if (remainder == 0) return 0;
  
  // Find largest divisor of remainder that's <= vectorLen
  for (int64_t stride = std::min(remainder, vectorLen); stride > 0; stride--) {
    if (remainder % stride == 0) {
      return stride;
    }
  }
  return 1; // Fallback to unroll one at a time
}

// Estimate instruction count for fully masked body with stride
// This estimates the number of instructions that will be generated
int64_t estimateMaskedBodyInstructions(int64_t m, int64_t n, int64_t k, int64_t stride) {
  // Outer loop: m iterations
  // Inner j loop: n/stride iterations (if stride divides n)
  // Per j iteration:
  //   - Mask creation: ~1 instruction
  //   - K loop: k iterations
  //     Per k iteration:
  //       - Scalar load from A: ~1 instruction
  //       - Broadcast: ~1 instruction
  //       - Masked load from B: ~1 instruction
  //       - FMA: ~1 instruction
  //       - Yield: ~1 instruction
  //   - Masked store: ~1 instruction
  // Total per j iteration: 1 (mask) + k * 5 (k-loop body) + 1 (store) = 2 + 5*k
  int64_t jIterations = n / stride; // stride divides n, so this is exact
  return m * jIterations * (2 + 5 * k);
}

// Estimate instruction count for masked remainder approach
int64_t estimateMaskedRemainderInstructions(int64_t m, int64_t n, int64_t k, int64_t vectorLen) {
  int64_t remainder = n % vectorLen;
  int64_t fullVectorIterations = n / vectorLen;
  
  // Main vectorized loop: m * fullVectorIterations iterations
  // Per iteration:
  //   - K loop: k iterations
  //     Per k iteration:
  //       - Scalar load: ~1
  //       - Broadcast: ~1
  //       - Vector load: ~1
  //       - FMA: ~1
  //       - Yield: ~1
  //   - Vector store: ~1
  // Total per main iteration: k * 5 + 1 = 5*k + 1
  
  // Remainder handling: m iterations
  // Per remainder iteration:
  //   - Mask creation: ~1
  //   - K loop: k iterations (same as above)
  //   - Masked store: ~1
  // Total per remainder iteration: 1 + 5*k + 1 = 2 + 5*k
  
  int64_t mainLoopInstrs = m * fullVectorIterations * (5 * k + 1);
  int64_t remainderInstrs = (remainder > 0) ? m * (2 + 5 * k) : 0;
  
  return mainLoopInstrs + remainderInstrs;
}

// Strategy function g(f(input)) - decides the best strategy
// Now with instruction count estimation to choose optimal stride
StrategyDecision determineStrategy(
    int64_t n, int64_t vectorLen, Type elementType,
    bool hasMaskingCapability, bool hasStaticDims,
    int64_t m = 0, int64_t k = 0) {
  
  StrategyDecision decision;
  InputWeights weights;
  CostEvaluation cost = evaluateCost(n, vectorLen, elementType, hasMaskingCapability, hasStaticDims);
  
  int64_t remainder = n % vectorLen;
  bool hasRemainder = (remainder != 0);
  
  // Get vector width to determine ISA (AVX vs AVX-512)
  // Note: vectorWidthOpt is a global variable, we need to access it
  // For now, we'll pass it as a parameter or check hasMaskingCapability
  // AVX-512 typically has better masking, so we can infer from that
  // But more accurately, we should check vectorWidthOpt >= 512
  
  // If no remainder, use regular vector operations (no masking overhead)
  // Perfect tiles should not use masking - it adds unnecessary overhead
  if (!hasRemainder) {
    decision.strategy = StrategyDecision::NO_MASKING;
    decision.stride = vectorLen;
    decision.useStride = false; // No stride needed - perfect alignment
    decision.cacheAligned = (getCacheLineStride(elementType, vectorLen) == vectorLen);
    decision.eliminatesRemainder = true;
    
    if (debugStrategyOpt) {
      llvm::errs() << "[Strategy Debug] Selected: NO_MASKING (perfect tile, no remainder)\n";
    }
    return decision;
  }
  
  // Simplified heuristic based on AVX/AVX2/AVX-512 benchmark results:
  // 1. If perfectly tiled (no remainder): Use NO_MASKING (already handled above)
  // 2. For AVX (128 bits): If remainder == 1, use UNROLL_REMAINDER
  // 3. For AVX2 (256 bits) and AVX-512 (512 bits): If remainder <= 3, use UNROLL_REMAINDER
  // 4. Otherwise: Use MASK_REMAINDER (masking wins for larger remainders)
  // Note: MASK_BODY is disabled in heuristic for now (code kept for future use)
  
  // Check if we should use unrolling based on remainder size and ISA
  bool shouldUnroll = false;
  if (remainder == 1) {
    // Remainder 1: Unrolling performs better for AVX, AVX2, and AVX-512
    shouldUnroll = true;
  } else if (remainder <= 3 && vectorWidthOpt >= 256) {
    // For AVX2 and AVX-512, unrolling wins for remainders 1-3
    shouldUnroll = true;
  }
  
  if (shouldUnroll) {
    // Unrolling performs better for small remainders
    decision.strategy = StrategyDecision::UNROLL_REMAINDER;
    decision.stride = findOptimalUnrollingStride(remainder, vectorLen);
    decision.useStride = (decision.stride > 1);
    decision.cacheAligned = false;
    decision.eliminatesRemainder = false;
    
    if (debugStrategyOpt) {
      if (vectorWidthOpt >= 512) {
        llvm::errs() << "[Strategy Debug] Selected: UNROLL_REMAINDER (remainder=" << remainder 
                     << " <= 3, AVX-512 unrolling wins)\n";
      } else if (vectorWidthOpt >= 256) {
        llvm::errs() << "[Strategy Debug] Selected: UNROLL_REMAINDER (remainder=" << remainder 
                     << " <= 3, AVX2 unrolling wins)\n";
      } else {
        llvm::errs() << "[Strategy Debug] Selected: UNROLL_REMAINDER (remainder=1, AVX unrolling wins)\n";
      }
    }
  } else if (hasMaskingCapability) {
    // Remainder > 1: Masking performs better
    decision.strategy = StrategyDecision::MASK_REMAINDER;
    decision.stride = vectorLen; // Use full vector for main loop
    decision.useStride = false;
    decision.cacheAligned = false;
    decision.eliminatesRemainder = false;
    
    if (debugStrategyOpt) {
      llvm::errs() << "[Strategy Debug] Selected: MASK_REMAINDER (remainder=" << remainder 
                   << ", masking wins)\n";
    }
  } else {
    // Masking not available, use unrolling
    decision.strategy = StrategyDecision::UNROLL_REMAINDER;
    decision.stride = findOptimalUnrollingStride(remainder, vectorLen);
    decision.useStride = (decision.stride > 1);
    decision.cacheAligned = false;
    decision.eliminatesRemainder = false;
    
    if (debugStrategyOpt) {
      llvm::errs() << "[Strategy Debug] Selected: UNROLL_REMAINDER (masking not available)\n";
    }
  }
  
  return decision;
}

// Pattern to convert MatmulOp to vector operations
struct MatmulToVectorPattern : public OpRewritePattern<linalg::MatmulOp> {
  MatmulToVectorPattern(MLIRContext *context, bool unrollScalarK, bool useMaskedRemainder, bool useBodyMasking)
      : OpRewritePattern<linalg::MatmulOp>(context),
        unrollScalarK(unrollScalarK),
        useMaskedRemainder(useMaskedRemainder),
        useFullyMasked(useBodyMasking) {}  // Keep internal name for now to minimize changes

  bool unrollScalarK;
  bool useMaskedRemainder;
  bool useFullyMasked;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    // Optional MLIR baseline: delegate to the generic linalg vectorizer.
    if (useMlirVectorizerOpt) {
      Operation *op = matmulOp.getOperation();

      // Quick pre-check: is there dedicated vectorization logic for this op?
      if (!linalg::hasVectorizationImpl(op))
        return failure();

      // Check vectorization preconditions; no explicit vector sizes for now.
      if (failed(linalg::vectorizeOpPrecondition(op)))
        return failure();

      auto resultOrErr = linalg::vectorize(rewriter, op);
      if (failed(resultOrErr))
        return failure();

      rewriter.eraseOp(op);
      return success();
    }

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
    
    // Determine strategy: use heuristic if enabled, otherwise use explicit flags
    StrategyDecision strategy;
    // AVX-512 has better masking, but AVX/AVX2 also support some masking
    // For heuristic, we allow masking for all vector widths
    bool hasMaskingCapability = (vectorWidthOpt >= 128); // AVX and above support masking
    
    if (useModelStrategyOpt) {
      // Use neural network model for strategy selection
      // Generate all possible (LS, remainder_strategy) combinations and pick best
      StrategyDecision best_strategy;
      float best_speedup = 0.0f;
      
      // Determine instruction type from vector width
      int instruction_type = 1; // Default AVX
      if (vectorWidthOpt >= 512) {
        instruction_type = 3; // AVX512
      } else if (vectorWidthOpt >= 256) {
        instruction_type = 2; // AVX2
      }
      
      // X * Y (repetition dimensions): For matmul, X = m, Y = 1
      int64_t X_times_Y = m;
      
      // K (vectorized dimension size) = n
      int64_t K = n;
      
      if (debugStrategyOpt) {
        llvm::errs() << "[Model Strategy] Evaluating configurations:\n";
        llvm::errs() << "  instruction_type=" << instruction_type 
                     << ", K=" << K << ", X_times_Y=" << X_times_Y << "\n";
      }
      
      // Track best configuration details
      int64_t best_LS = vectorLen;
      int64_t best_remainder_strategy = 0;
      
      // Generate all possible configurations
      for (int64_t LS = 1; LS <= vectorLen; ++LS) {
        int64_t LS_equals_VS = (LS == vectorLen) ? 1 : 0;
        int64_t K_remainder = K % LS;  // Numeric remainder, not binary
        int64_t LS_div_K = (K > 0) ? (LS / K) : 0;  // LS // K
        
        // Test both remainder strategies (masking=0, unrolling=1)
        for (int64_t remainder_strategy = 0; remainder_strategy <= 1; ++remainder_strategy) {
          // Predict speedup for this configuration
          float predicted_speedup = predictSpeedupFromFeatures(
            instruction_type, LS, LS_equals_VS, K_remainder,
            remainder_strategy, X_times_Y, LS_div_K
          );
          
          if (debugStrategyOpt) {
            llvm::errs() << "  LS=" << LS 
                         << ", remainder_strategy=" << (remainder_strategy == 0 ? "masking" : "unrolling")
                         << ", predicted_speedup=" << predicted_speedup << "x\n";
          }
          
          // Keep track of best
          if (predicted_speedup > best_speedup) {
            best_speedup = predicted_speedup;
            best_LS = LS;
            best_remainder_strategy = remainder_strategy;
            
            // Map to StrategyDecision
            // LS = Logical Size (masking stride for body)
            // VS = Vector Size (vectorLen)
            // When LS < VS: Body is masked with stride LS, remainder still needs handling
            // remainder_strategy: 0=masking, 1=unrolling for the remainder part
            if (K_remainder == 0 && LS == vectorLen) {
              // Perfect alignment (K % LS == 0) and using full vector width - no masking needed
              best_strategy.strategy = StrategyDecision::NO_MASKING;
              best_strategy.stride = vectorLen;
              best_strategy.useStride = false;
              best_strategy.eliminatesRemainder = true;
            } else if (LS < vectorLen) {
              // LS < VS: Body is masked with stride LS (MASK_BODY)
              // Remainder handling depends on remainder_strategy:
              // - remainder_strategy=0: mask the remainder too (fully masked)
              // - remainder_strategy=1: unroll the remainder
              best_strategy.strategy = StrategyDecision::MASK_BODY;
              best_strategy.stride = LS;
              best_strategy.useStride = true;
              best_strategy.eliminatesRemainder = (K % LS == 0);
              // Store remainder strategy info: we'll use MASK_BODY and handle remainder
              // The remainder_strategy is already captured in best_remainder_strategy
            } else if (LS == vectorLen && remainder_strategy == 0) {
              // LS == VS with masking means mask only the remainder - this is MASK_REMAINDER
              best_strategy.strategy = StrategyDecision::MASK_REMAINDER;
              best_strategy.stride = vectorLen;
              best_strategy.useStride = false;
              best_strategy.eliminatesRemainder = false;
            } else {
              // LS == VS with unrolling: unroll the remainder
              best_strategy.strategy = StrategyDecision::UNROLL_REMAINDER;
              best_strategy.stride = vectorLen;
              best_strategy.useStride = false;
              best_strategy.eliminatesRemainder = false;
            }
            
            // Check cache alignment
            int64_t cacheLineStride = getCacheLineStride(elementType, vectorLen);
            best_strategy.cacheAligned = (LS == cacheLineStride);
          }
        }
      }
      
      strategy = best_strategy;
      
      if (debugStrategyOpt) {
        const char* strategy_name = "UNKNOWN";
        switch (best_strategy.strategy) {
          case StrategyDecision::NO_MASKING:
            strategy_name = "NO_MASKING";
            break;
          case StrategyDecision::MASK_REMAINDER:
            strategy_name = "MASK_REMAINDER";
            break;
          case StrategyDecision::UNROLL_REMAINDER:
            strategy_name = "UNROLL_REMAINDER";
            break;
          case StrategyDecision::MASK_BODY:
            strategy_name = "MASK_BODY";
            break;
        }
        const char* remainder_strategy_name = (best_remainder_strategy == 0) ? "masking" : "unrolling";
        llvm::errs() << "[Model Strategy] Selected: " << strategy_name;
        
        if (best_LS < vectorLen) {
          // LS < VS: Body is masked with stride LS, remainder handled separately
          llvm::errs() << " (body_masked=true, body_stride=LS=" << best_LS
                       << ", remainder=" << remainder_strategy_name << ")";
        } else {
          // LS == VS: No body masking, only remainder handling
          llvm::errs() << " (body_masked=false, remainder=" << remainder_strategy_name << ")";
        }
        llvm::errs() << " [predicted_speedup=" << best_speedup << "x]\n";
      }
    } else if (useHeuristicStrategyOpt) {
      // Use heuristic system g(f(input))
      // Inputs: alignment/vector size, ISA masking capability, cache line size
      // Pass m and k dimensions for accurate instruction counting
      strategy = determineStrategy(
          n, vectorLen, elementType, hasMaskingCapability, hasStaticDims, m, kDim);
    } else {
      // Use explicit flags (backward compatibility / blind strategies)
      strategy = StrategyDecision();
      // Check if body masking stride is set via explicit flags
      int64_t bodyStride = bodyMaskingStrideOpt;
      if (bodyStride > 0 && bodyStride < vectorLen) {
        strategy.strategy = StrategyDecision::MASK_BODY;
        strategy.stride = bodyStride;
        strategy.useStride = false;
      } else if (useMaskedRemainder) {
        strategy.strategy = StrategyDecision::MASK_REMAINDER;
        strategy.stride = vectorLen;
        strategy.useStride = false;
      } else if (unrollScalarK) {
        strategy.strategy = StrategyDecision::UNROLL_REMAINDER;
        strategy.stride = 1;
        strategy.useStride = false;
      } else {
        // Default: scalar remainder (no special strategy)
        strategy.strategy = StrategyDecision::UNROLL_REMAINDER;
        strategy.stride = 1;
        strategy.useStride = false;
      }
    }

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
    auto buildScalarK = [&](Value iIdx, Value jIdx, bool shouldUnrollK = false) -> Value {
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
      // Also enable if heuristic chose UNROLL_REMAINDER
      bool canFullyUnroll =
          (unrollScalarK || shouldUnrollK) && !lhsType.isDynamicDim(1) && kDim > 0;

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
    // Use heuristic-determined strategy, or fall back to explicit flags
    // Body masking is used when body masking stride is set AND LS < VS
    // If LS == VS, it's no masking (use full vector width)
    // Note: body masking stride and remainder handling are independent
    int64_t bodyStride = bodyMaskingStrideOpt;
    bool shouldUseBodyMasking = (strategy.strategy == StrategyDecision::MASK_BODY) || 
                               (bodyStride > 0 && bodyStride < vectorLen);
    bool shouldUseNoMasking = (strategy.strategy == StrategyDecision::NO_MASKING);
    
    // For perfect tiles (no remainder), use regular vector operations without masking
    if (shouldUseNoMasking) {
      // Skip the fully masked path and go straight to regular vectorized loops below
      // This avoids masking overhead for perfect tiles
    } else if (shouldUseBodyMasking) {
      // Use heuristic-determined stride, or override with explicit options
      int64_t effectiveStride = strategy.useStride ? strategy.stride : vectorLen;
      bool useCustomStride = strategy.useStride;
      
      // Allow explicit override via command-line options (for backward compatibility)
      int64_t cacheLineStride = 0;
      if (useCacheLineStrideOpt) {
        cacheLineStride = getCacheLineStride(elementType, vectorLen);
      }
      
      // Get body masking stride (LS) - check both new and legacy option names
      int64_t bodyStride = bodyMaskingStrideOpt;
      
      if (bodyStride > 0) {
        // When LS < VS: Body is masked with stride LS
        // Remainder handling is independent (handled separately below)
        if (bodyStride <= vectorLen) {
          effectiveStride = bodyStride;
          useCustomStride = true;
          
          if (debugStrategyOpt) {
            llvm::errs() << "[Body Masking] Using body stride LS=" << bodyStride 
                         << " (VS=" << vectorLen << ", n=" << n 
                         << ", remainder=" << (n % bodyStride) << ")\n";
          }
        } else {
          if (debugStrategyOpt) {
            llvm::errs() << "[Body Masking] Warning: bodyStride=" << bodyStride 
                         << " > VS=" << vectorLen << ", ignoring\n";
          }
        }
      } else if (useCacheLineStrideOpt && cacheLineStride > 0) {
        if (hasStaticDims) {
          if (n % cacheLineStride == 0 && cacheLineStride <= vectorLen) {
            effectiveStride = cacheLineStride;
            useCustomStride = true;
          }
        } else {
          if (cacheLineStride <= vectorLen) {
            effectiveStride = cacheLineStride;
            useCustomStride = true;
          }
        }
      } else if (strategy.useStride) {
        // Use heuristic-determined stride
        effectiveStride = strategy.stride;
        useCustomStride = true;
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
          
          int64_t bodyStride = bodyMaskingStrideOpt;
          if (bodyStride > 0 && bodyStride <= vectorLen) {
            strideCandidate = rewriter.create<arith::ConstantIndexOp>(loc, bodyStride);
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
      
      // When LS < VS: Split into BODY loop (masked with stride LS) and REMAINDER loop (handled separately)
      // Calculate body loop end: (n / LS) * LS (no remainder)
      Value bodyLoopEnd;
      if (hasStaticDims && useCustomStride) {
        int64_t bodyLoopEndStatic = (n / effectiveStride) * effectiveStride;
        bodyLoopEnd = rewriter.create<arith::ConstantIndexOp>(loc, bodyLoopEndStatic);
      } else {
        // Dynamic: bodyLoopEnd = (n / finalStride) * finalStride
        auto nDivStride = rewriter.create<arith::DivUIOp>(loc, nConst, finalStride);
        bodyLoopEnd = rewriter.create<arith::MulIOp>(loc, nDivStride, finalStride);
      }
      
      // BODY LOOP: From 0 to bodyLoopEnd with step LS (all fully masked, no remainder)
      auto bodyLoop = rewriter.create<scf::ForOp>(loc, zero, bodyLoopEnd, finalStride);
      rewriter.setInsertionPointToStart(bodyLoop.getBody());
      Value jBody = bodyLoop.getInductionVar();
      
      // Body loop: always fully masked with stride LS (no remainder to handle)
      auto maskTypeBody = hasStaticDims && useCustomStride 
          ? VectorType::get({effectiveStride}, rewriter.getI1Type())
          : VectorType::get({vectorLen}, rewriter.getI1Type());
      // Create full mask (all elements active in body)
      Value fullMaskCount = finalStride;
      Value maskBody = rewriter.create<vector::CreateMaskOp>(loc, maskTypeBody, fullMaskCount);
      
      // Zero vector with effective vector type
      auto zeroElement = rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getZeroAttr(elementType));
      auto zeroVec = rewriter.create<vector::BroadcastOp>(loc, effectiveVectorType, zeroElement);
      
      // K-loop for body
      SmallVector<Value> initArgsBody{zeroVec};
      auto kLoopBody = rewriter.create<scf::ForOp>(loc, zero, kConst, one, initArgsBody);
      rewriter.setInsertionPointToStart(kLoopBody.getBody());
      Value kBody = kLoopBody.getInductionVar();
      Value accVecBody = kLoopBody.getBody()->getArgument(1);
      
      // Load and broadcast A[i,k]
      SmallVector<Value> lhsIdx{i, kBody};
      auto aScalarBody = rewriter.create<memref::LoadOp>(loc, elementType, lhs, lhsIdx);
      auto aVecBody = rewriter.create<vector::BroadcastOp>(loc, effectiveVectorType, aScalarBody);
      
      // Masked load B[k, j:j+finalStride] (full mask in body)
      SmallVector<Value> rhsIdx{kBody, jBody};
      auto bVecBody = rewriter.create<vector::MaskedLoadOp>(
          loc, effectiveVectorType, rhs, rhsIdx, maskBody, zeroVec);
      
      // FMA
      auto fmaResultBody = rewriter.create<vector::FMAOp>(loc, effectiveVectorType, aVecBody, bVecBody, accVecBody);
      rewriter.create<scf::YieldOp>(loc, ValueRange{fmaResultBody.getResult()});
      
      // Masked store
      rewriter.setInsertionPointAfter(kLoopBody);
      Value finalVecBody = kLoopBody.getResults()[0];
      SmallVector<Value> resultIdxBody{i, jBody};
      rewriter.create<vector::MaskedStoreOp>(loc, result, resultIdxBody, maskBody, finalVecBody);
      
      // Move insertion point after body loop
      rewriter.setInsertionPointAfter(bodyLoop);
      
      // REMAINDER LOOP: From bodyLoopEnd to n (handled separately based on remainder_strategy)
      // Check if there's a remainder
      bool hasRemainderAtCompileTime = hasStaticDims && (n % effectiveStride != 0);
      if (hasRemainderAtCompileTime || !hasStaticDims) {
        // Determine remainder handling strategy
        bool shouldUseMaskedRemainder = useMaskedRemainderOpt;
        bool shouldUnrollRemainder = unrollScalarKOpt && !shouldUseMaskedRemainder;
        
        if (debugStrategyOpt && hasStaticDims) {
          int64_t remainder = n % effectiveStride;
          const char* remainderMethod = shouldUnrollRemainder ? "unrolling" : 
                                       (shouldUseMaskedRemainder ? "masking" : "masking (default)");
          llvm::errs() << "[Body Masking] Body loop: 0 to " << (n / effectiveStride) * effectiveStride 
                       << " step " << effectiveStride << " (fully masked)\n";
          llvm::errs() << "[Body Masking] Remainder: " << (n / effectiveStride) * effectiveStride 
                       << " to " << n << " (remainder=" << remainder 
                       << ", handled by " << remainderMethod << ")\n";
        }
        
        if (shouldUseMaskedRemainder) {
          // Masked remainder: use buildMaskedRemainder helper
          buildMaskedRemainder(i, bodyLoopEnd, nConst);
        } else {
          // Unrolled remainder: scalar loop
          bool unrollK = unrollScalarKOpt;
          auto remLoop = rewriter.create<scf::ForOp>(loc, bodyLoopEnd, nConst, one);
          rewriter.setInsertionPointToStart(remLoop.getBody());
          Value jRem = remLoop.getInductionVar();
          Value finalScalar = buildScalarK(i, jRem, unrollK);
          rewriter.create<memref::StoreOp>(loc, finalScalar, result, ValueRange{i, jRem});
        }
      }
      
      // outerLoopI also doesn't need a yield since it has no iter_args
      rewriter.setInsertionPointAfter(outerLoopI);
      rewriter.eraseOp(matmulOp);
      return success();
    }

    // For NO_MASKING strategy (perfect tiles), skip masking and use regular vector ops
    // This falls through to the regular vectorized loops below (no masking overhead)

    // Create loops for vectorized matmul (regular, no masking)
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

    // Choose remainder strategy based on heuristic decision (declare before goto)
    bool shouldUseMaskedRemainder = (strategy.strategy == StrategyDecision::MASK_REMAINDER) || useMaskedRemainder;
    bool shouldUnrollRemainder = (strategy.strategy == StrategyDecision::UNROLL_REMAINDER) || 
                                  (unrollScalarK && !shouldUseMaskedRemainder);
    
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
    
    if (shouldUseMaskedRemainder) {
      buildMaskedRemainder(i, fullVectorCount, nConst);
    } else if (shouldUnrollRemainder && strategy.useStride && strategy.stride > 1 && !useHeuristicStrategyOpt) {
      // Unroll remainder with optimal stride (only for explicit flags, not heuristic)
      // Note: Stride unrolling is disabled for heuristic in this stage
      auto strideConst = rewriter.create<arith::ConstantIndexOp>(loc, strategy.stride);
      auto remLoop = rewriter.create<scf::ForOp>(loc, fullVectorCount, nConst, strideConst);
      rewriter.setInsertionPointToStart(remLoop.getBody());
      Value jRem = remLoop.getInductionVar();
      
      // Unroll stride iterations
      // Enable k-loop unrolling if explicit flag is set
      bool unrollK = unrollScalarK;
      for (int64_t offset = 0; offset < strategy.stride; offset++) {
        Value offsetConst = rewriter.create<arith::ConstantIndexOp>(loc, offset);
        Value jOffset = rewriter.create<arith::AddIOp>(loc, jRem, offsetConst);
        // Check bounds
        auto inBounds = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, jOffset, nConst);
        auto ifInBounds = rewriter.create<scf::IfOp>(loc, inBounds, false);
        rewriter.setInsertionPointToStart(ifInBounds.thenBlock());
        Value finalScalar = buildScalarK(i, jOffset, unrollK);
        rewriter.create<memref::StoreOp>(loc, finalScalar, result, ValueRange{i, jOffset});
        rewriter.setInsertionPointAfter(ifInBounds);
      }
    } else {
      // Standard scalar remainder loop
      // Enable k-loop unrolling if explicit flag is set OR heuristic chose UNROLL_REMAINDER
      bool unrollK = unrollScalarK || (useHeuristicStrategyOpt && strategy.strategy == StrategyDecision::UNROLL_REMAINDER);
      auto remLoop = rewriter.create<scf::ForOp>(loc, fullVectorCount, nConst, one);
      rewriter.setInsertionPointToStart(remLoop.getBody());
      Value jRem = remLoop.getInductionVar();
      Value finalScalar = buildScalarK(i, jRem, unrollK);
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
    // Body masking is used if body masking stride is set AND LS < VS
    // Body masking and remainder handling are independent
    // Note: We can't check vectorLen here without element type, so we check in the pattern
    // If bodyStride > 0, the pattern will check if it's < VS
    int64_t bodyStride = bodyMaskingStrideOpt;
    bool useBodyMasking = (bodyStride > 0);
        
    patterns.add<MatmulToVectorPattern>(context, shouldUnroll, useMaskedRemainder, useBodyMasking);

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
