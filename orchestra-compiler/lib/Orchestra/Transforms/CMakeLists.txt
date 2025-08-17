add_library(OrchestraTransforms
  SpeculateIfOp.cpp
  Passes.cpp
  LowerOrchestraToStandard.cpp
  LowerOrchestraToGPU.cpp
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
