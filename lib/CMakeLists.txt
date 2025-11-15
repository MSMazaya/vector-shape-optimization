add_mlir_library(MLIRVectorShapeOpt
  PrintLinalgOps.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRPass
)

target_include_directories(MLIRVectorShapeOpt
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
