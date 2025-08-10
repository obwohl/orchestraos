add_executable(orchestra-opt main.cpp)

target_include_directories(orchestra-opt PUBLIC
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)

target_link_directories(orchestra-opt PUBLIC
  ${PROJECT_BINARY_DIR}/lib/Orchestra
  ${PROJECT_BINARY_DIR}/lib/OrchestraPasses
  ${MLIR_LIBRARY_DIRS}
  ${LLVM_LIBRARY_DIRS}
)

target_link_libraries(orchestra-opt PUBLIC
  MLIRMlirOptMain
  MLIRFuncDialect
  Orchestra
  OrchestraPasses
  ${MLIR_LIBRARIES}
  ${LLVM_LIBRARIES}
)