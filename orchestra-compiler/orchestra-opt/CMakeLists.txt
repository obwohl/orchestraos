add_executable(orchestra-opt main.cpp)

target_include_directories(orchestra-opt PUBLIC
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(orchestra-opt PUBLIC
  MLIRMlirOptMain
  MLIRFuncDialect
  # Force the linker to include all object files from the Orchestra
  # static library to ensure static initializers for op registration are run.
  # This is crucial for out-of-tree dialects.
  "-Wl,--whole-archive"
  Orchestra
  "-Wl,--no-whole-archive"
  ${MLIR_LIBRARIES}
  ${LLVM_LIBRARIES}
)
