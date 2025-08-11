# This file's sole responsibility is code generation.
# It defines the custom target that runs mlir-tblgen on OrchestraOps.td.
# The first argument, "OrchestraOps", is the basename of the .td file.
# The second argument, "orchestra", is the C++ namespace for the dialect.
# This command creates the target "MLIROrchestraOpsIncGen".
add_mlir_dialect(OrchestraOps orchestra)
