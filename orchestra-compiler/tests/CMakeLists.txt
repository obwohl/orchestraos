# Find the FileCheck executable
find_program(LLVM_FILECHECK_EXECUTABLE FileCheck-20
  HINTS ${LLVM_TOOLS_DIR})
if(NOT LLVM_FILECHECK_EXECUTABLE)
    message(FATAL_ERROR "FileCheck not found")
endif()

# Define the path to the orchestra-opt executable
set(ORCHESTRA_OPT_EXECUTABLE ${CMAKE_BINARY_DIR}/orchestra-opt/orchestra-opt)

# Define the test source root
set(TEST_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Configure the lit configuration file.
# This generates lit.cfg.py in the build directory.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py
    @ONLY
)

# Add the lit test suite.
# The DEPENDS keyword is crucial to ensure orchestra-opt is built before the tests are run.
add_lit_testsuite(
    check-orchestra
    "Running the Orchestra regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
      orchestra-opt
)
