#ifndef MY_PASSES_H
#define MY_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
std::unique_ptr<Pass> createPrintLinalgOpsPass();
}

#endif
