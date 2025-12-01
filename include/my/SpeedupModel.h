// SpeedupModel.h
// Header for ONNX-based speedup prediction model

#ifndef SPEEDUP_MODEL_H
#define SPEEDUP_MODEL_H

#include <string>
#include <cstdint>

// Initialize the speedup prediction model
// Returns true if successful, false otherwise
bool initializeSpeedupModel(const std::string& model_path);

// Predict speedup from features
// Input features:
//   instruction_type: 1=AVX, 2=AVX2, 3=AVX512, etc.
//   LS: Logical Size [1, VS]
//   LS_equals_VS: 1 if LS == VS, else 0
//   K_remainder: K % LS (numeric remainder value)
//   remainder_strategy: 0=masking, 1=unrolling
//   X_times_Y: Product of repetition dimensions
//   LS_div_K: LS // K (integer division)
// Returns: Predicted speedup (scalar_time / vectorized_time)
float predictSpeedupFromFeatures(int64_t instruction_type, int64_t LS, int64_t LS_equals_VS,
                                  int64_t K_remainder, int64_t remainder_strategy,
                                  int64_t X_times_Y, int64_t LS_div_K);

#endif  // SPEEDUP_MODEL_H

