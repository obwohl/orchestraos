add_library(OrchestraTransforms
  SpeculateIfOp.cpp
  Passes.cpp
)

target_link_libraries(OrchestraTransforms
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRAnalysis
  MLIRSCFDialect
  MLIRSupport
  LLVMSupport
  Orchestra
)

add_dependencies(OrchestraTransforms Orchestra)
