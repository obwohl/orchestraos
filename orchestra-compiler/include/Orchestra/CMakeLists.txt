add_subdirectory(Transforms)
add_subdirectory(Rock)

add_mlir_dialect(Orchestra
  Orchestra.td
  OrchestraDialect.td
  OrchestraInterfaces.td
  OrchestraOps.td
)
