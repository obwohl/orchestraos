add_mlir_dialect_library(Orchestra
  OrchestraDialect.cpp
  OrchestraOps.cpp

  DEPENDS
  OrchestraIncGen

  LINK_LIBS
  PUBLIC
  MLIRIR
)
