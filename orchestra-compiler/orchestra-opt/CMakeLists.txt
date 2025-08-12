# Create the orchestra-opt executable.
add_executable(orchestra-opt
  main.cpp
  )

# Link the executable against the Orchestra dialect's object files and the
# necessary MLIR libraries for a basic '-opt' style tool.
target_link_libraries(orchestra-opt PRIVATE
  # By using $<TARGET_OBJECTS:Orchestra>, we link the compiled .o files
  # directly into the executable, completely bypassing the static library
  # symbol elision problem.
  $<TARGET_OBJECTS:Orchestra>

  # Link against the necessary MLIR libraries.
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRDialect
  MLIRParser
  MLIRMlirOptMain
)
