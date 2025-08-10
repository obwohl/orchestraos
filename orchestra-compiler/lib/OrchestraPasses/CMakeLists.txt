add_library(OrchestraPasses
  DivergenceToSpeculation.cpp
  DummyPass.cpp
  OrchestraPasses.cpp
)

target_link_libraries(OrchestraPasses PRIVATE Orchestra)
