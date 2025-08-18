add_library(OrchestraTransforms
  SpeculateIfOp.cpp
  Passes.cpp
  LowerOrchestraToStandard.cpp
  LowerOrchestraToGPU.cpp
  LowerOrchestraToXeGPU.cpp
)

target_link_libraries(OrchestraTransforms
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRAnalysis
  MLIRSCFDialect
  MLIRNVGPUDialect
  MLIRSupport
  LLVMSupport
  Orchestra
)

add_dependencies(OrchestraTransforms Orchestra)
