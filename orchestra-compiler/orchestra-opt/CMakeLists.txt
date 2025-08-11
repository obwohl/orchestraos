# Create the orchestra-opt executable.
add_executable(orchestra-opt
  main.cpp
  )

# Link the executable against the Orchestra dialect library and the
# necessary MLIR libraries for a basic '-opt' style tool.
target_link_libraries(orchestra-opt PRIVATE
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRDialect
  MLIRParser
  MLIRMlirOptMain

  # Force the linker to include all object files from the Orchestra
  # static library to ensure static initializers for op registration are run.
  "-Wl,--whole-archive"
  Orchestra
  "-Wl,--no-whole-archive"
  )
