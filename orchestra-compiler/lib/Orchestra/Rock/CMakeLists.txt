add_mlir_dialect_library(OrchestraRock
  RockDialect.cpp
  RockOps.cpp
  DEPENDS
  RockIncGen
)
