//===- handle-sopp.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handlers.h"

using namespace llvm;

namespace COMGR::hotswap {

Expected<HandlerResult> handleSOPP(RaiseContext &Ctx, const DecodedInst &Di,
                                   OpResolver &Op) {
  (void)Op;
  HandlerResult Hr;

  // s_endpgm is the only SOPP the raiser lifts so far: it terminates the
  // kernel's control flow with a return. Every other SOPP (branches, waits,
  // barriers, ...) refuses until its handler lands.
  if (Di.CanonOp == CanonicalOp::S_ENDPGM) {
    Ctx.B.CreateRetVoid();
    Hr.Handled = true;
    return Hr;
  }

  return RaiseFailure::unsupportedInstructionForm(
      strippedMnemonic(Ctx.Mc, Di.Inst), Di.Offset, "SOPP");
}

} // namespace COMGR::hotswap
