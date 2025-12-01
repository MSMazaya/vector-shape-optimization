add_mlir_library(MLIRVectorShapeOpt
  PrintLinalgOps.cpp
  LinalgToVector.cpp
  SpeedupModelSimple.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRVectorDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRPass
    MLIRTransformUtils
)

target_include_directories(MLIRVectorShapeOpt
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
