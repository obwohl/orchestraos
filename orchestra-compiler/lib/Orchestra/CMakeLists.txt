add_library(Orchestra
  OrchestraDialect.cpp
  "${PROJECT_BINARY_DIR}/include/Orchestra/OrchestraOps.cpp.inc"
)

set_target_properties(Orchestra PROPERTIES LINK_WHOLE_ARCHIVE ON)

target_link_libraries(Orchestra
  PUBLIC
  MLIRIR
  MLIRSupport
  MLIRParser
)

add_dependencies(Orchestra MLIROrchestraOpsIncGen)
