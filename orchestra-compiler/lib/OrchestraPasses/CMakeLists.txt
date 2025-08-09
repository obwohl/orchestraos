add_library(OrchestraPasses
  DivergenceToSpeculation.cpp
)

target_link_libraries(OrchestraPasses PRIVATE Orchestra)
