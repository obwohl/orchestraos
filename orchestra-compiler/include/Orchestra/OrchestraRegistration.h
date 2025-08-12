#ifndef ORCHESTRA_ORCHESTRAREGISTRATION_H
#define ORCHESTRA_ORCHESTRAREGISTRATION_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

// This function provides an explicit hook for the main executable to call to
// register the Orchestra dialect. It is marked 'extern "C"' to ensure a stable,
// unmangled name that can be easily linked from the main executable.
extern "C" void registerOrchestraDialect(mlir::DialectRegistry &registry);

#endif // ORCHESTRA_ORCHESTRAREGISTRATION_H
