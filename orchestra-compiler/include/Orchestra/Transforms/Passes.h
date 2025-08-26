//===- Passes.h - Orchestra pass entry points -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ORCHESTRA_TRANSFORMS_PASSES_H
#define ORCHESTRA_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createLowerOrchestraToStandardPass();
std::unique_ptr<Pass> createLowerOrchestraToGPU();
std::unique_ptr<Pass> createLowerOrchestraToROCDL();
std::unique_ptr<Pass> createLowerOrchestraToXeGPU();
std::unique_ptr<Pass> createLowerLinalgToRockPass();

void registerLoweringToStandardPasses();
void registerLoweringToGPUPasses();
void registerLoweringToROCDLPasses();
void registerLoweringToXeGPUPasses();
void registerLowerLinalgToRockPasses();


//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Orchestra/Transforms/Passes.h.inc"

} // end namespace mlir

#endif // ORCHESTRA_TRANSFORMS_PASSES_H
