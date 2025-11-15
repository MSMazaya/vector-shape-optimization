#include "my/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct PrintLinalgOpsPass
    : public PassWrapper<PrintLinalgOpsPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintLinalgOpsPass)

  StringRef getArgument() const final { return "print-linalg-ops"; }

  StringRef getDescription() const final {
    return "Print all linalg.* operations inside a function";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    llvm::outs() << "=== Function: " << func.getName() << " ===\n";

    func.walk([](Operation *op) {
      auto *dialect = op->getDialect();
      if (!dialect)
        return;
      if (dialect->getNamespace() != "linalg")
        return;
      llvm::outs() << "Found linalg op: " << op->getName() << "\n";
      op->print(llvm::outs());
      llvm::outs() << "\n\n";
    });
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createPrintLinalgOpsPass() {
  return std::make_unique<PrintLinalgOpsPass>();
}
} // namespace mlir

static PassRegistration<PrintLinalgOpsPass> PrintLinalgOpsPassReg;
