#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "my/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;

  registry.insert<
      func::FuncDialect,
      linalg::LinalgDialect,
      tensor::TensorDialect,
      scf::SCFDialect,
      arith::ArithDialect,
      math::MathDialect>();

  registerPass([]() -> std::unique_ptr<Pass> {
    return createPrintLinalgOpsPass();
  });

  return failed(MlirOptMain(argc, argv,
                            "Vector shape optimizer\n",
                            registry));
}
