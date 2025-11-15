get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport

  MLIRFuncDialect
  MLIRLinalgDialect

  MLIRVectorShapeOpt

  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
)

add_mlir_tool(vector-shape-opt
  vector-shape-opt.cpp
)

llvm_update_compile_flags(vector-shape-opt)

target_link_libraries(vector-shape-opt
  PRIVATE
    ${LIBS}
    MLIROptLib
)
