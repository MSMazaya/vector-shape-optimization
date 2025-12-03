// SpeedupModelSimple.cpp
// Simple C++ implementation of neural network for speedup prediction
// No external dependencies - pure C++ implementation

#include "my/SpeedupModel.h"
#include "my/SpeedupModelWeights.h"

#include <vector>
#include <cmath>
#include <algorithm>

namespace {

static bool g_model_loaded = true;  // Model is statically embedded

// ReLU activation
inline float relu(float x) {
  return std::max(0.0f, x);
}

// Helper macro to implement prediction for a specific ISA namespace
#define IMPLEMENT_PREDICT_SPEEDUP(ISA_NS) \
float predictSpeedup##ISA_NS(int64_t LS, int64_t LS_equals_VS, int64_t K_remainder, \
                              int64_t remainder_strategy, int64_t X_times_Y, int64_t LS_div_K) { \
  using namespace SpeedupModelWeights##ISA_NS; \
  std::vector<float> features = { \
    static_cast<float>(LS), static_cast<float>(LS_equals_VS), static_cast<float>(K_remainder), \
    static_cast<float>(remainder_strategy), static_cast<float>(X_times_Y), static_cast<float>(LS_div_K) \
  }; \
  for (size_t i = 0; i < features.size(); ++i) { \
    features[i] = (features[i] - FEATURE_MEAN[i]) / FEATURE_SCALE[i]; \
  } \
  std::vector<float> nonlin_features(features.begin(), features.begin() + 5); \
  float x_times_y = features[5]; \
  std::vector<float> h1(64); \
  for (int i = 0; i < 64; ++i) { \
    float sum = B1[i]; \
    for (int j = 0; j < 5; ++j) sum += W1[i][j] * nonlin_features[j]; \
    h1[i] = relu(sum); \
  } \
  std::vector<float> h2(32); \
  for (int i = 0; i < 32; ++i) { \
    float sum = B2[i]; \
    for (int j = 0; j < 64; ++j) sum += W2[i][j] * h1[j]; \
    h2[i] = relu(sum); \
  } \
  std::vector<float> h3(16); \
  for (int i = 0; i < 16; ++i) { \
    float sum = B3[i]; \
    for (int j = 0; j < 32; ++j) sum += W3[i][j] * h2[j]; \
    h3[i] = relu(sum); \
  } \
  float linear_out = LINEAR_WEIGHT * x_times_y; \
  std::vector<float> combined(17); \
  for (int i = 0; i < 16; ++i) combined[i] = h3[i]; \
  combined[16] = linear_out; \
  std::vector<float> out1(8); \
  for (int i = 0; i < 8; ++i) { \
    float sum = B_OUT1[i]; \
    for (int j = 0; j < 17; ++j) sum += W_OUT1[i][j] * combined[j]; \
    out1[i] = relu(sum); \
  } \
  float speedup = B_OUT2[0]; \
  for (int i = 0; i < 8; ++i) speedup += W_OUT2[0][i] * out1[i]; \
  return std::max(0.1f, std::min(50.0f, speedup)); \
}

// Implement for each ISA
IMPLEMENT_PREDICT_SPEEDUP(AVX)
IMPLEMENT_PREDICT_SPEEDUP(AVX2)
IMPLEMENT_PREDICT_SPEEDUP(AVX512)
IMPLEMENT_PREDICT_SPEEDUP(SVE)

#undef IMPLEMENT_PREDICT_SPEEDUP


// Predict speedup using the neural network, selecting model based on instruction_type
float predictSpeedupInternal(int64_t instruction_type, int64_t LS, int64_t LS_equals_VS,
                              int64_t K_remainder, int64_t remainder_strategy,
                              int64_t X_times_Y, int64_t LS_div_K) {
  if (!g_model_loaded) {
    return 1.0f;  // Fallback if model not loaded
  }
  
  // Select model based on instruction_type (ISA)
  // 1=AVX, 2=AVX2, 3=AVX512, 4=SVE
  switch (instruction_type) {
    case 1:  // AVX
      return predictSpeedupAVX(LS, LS_equals_VS, K_remainder, remainder_strategy, X_times_Y, LS_div_K);
    case 2:  // AVX2
      return predictSpeedupAVX2(LS, LS_equals_VS, K_remainder, remainder_strategy, X_times_Y, LS_div_K);
    case 3:  // AVX512
      return predictSpeedupAVX512(LS, LS_equals_VS, K_remainder, remainder_strategy, X_times_Y, LS_div_K);
    case 4:  // SVE
      return predictSpeedupSVE(LS, LS_equals_VS, K_remainder, remainder_strategy, X_times_Y, LS_div_K);
    default:
      // Fallback to AVX if unknown
      return predictSpeedupAVX(LS, LS_equals_VS, K_remainder, remainder_strategy, X_times_Y, LS_div_K);
  }
}

}  // namespace

// Public API

bool initializeSpeedupModel(const std::string& model_path) {
  // Model is statically embedded, always loaded
  g_model_loaded = true;
  return true;
}

float predictSpeedupFromFeatures(int64_t instruction_type, int64_t LS, int64_t LS_equals_VS,
                                  int64_t K_remainder, int64_t remainder_strategy,
                                  int64_t X_times_Y, int64_t LS_div_K) {
  return predictSpeedupInternal(instruction_type, LS, LS_equals_VS, K_remainder,
                                remainder_strategy, X_times_Y, LS_div_K);
}

