#ifndef ORCHESTRA_REGISTRATION_H
#define ORCHESTRA_REGISTRATION_H

namespace orchestra {
// This function should be called from the main function of any executable
// that needs to use the Orchestra dialect. It will ensure that the
// dialect's operations are registered with the MLIR context.
void ensureOrchestraDialectRegistered();
} // namespace orchestra

#endif // ORCHESTRA_REGISTRATION_H
