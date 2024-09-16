//===-- RegAllocBasic.cpp - Basic Register Allocator ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RABitEA function pass, which provides a minimal
// implementation of the basic register allocator.
//
//===----------------------------------------------------------------------===//

#include "AllocationOrder.h"
#include "RegAllocBitEA.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>

#include "mt-BitEA/mtBitEA.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"


static RegisterRegAlloc biteaRegAlloc("bitea", "bitea register allocator",
                                      createBitEARegisterAllocator);

char &llvm::RABitEAID = RABitEA::ID;

INITIALIZE_PASS_BEGIN(RABitEA, "regallocbitea", "BitEA Register Allocator",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariables)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(RegisterCoalescer)
INITIALIZE_PASS_DEPENDENCY(MachineScheduler)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_END(RABitEA, "regallocbitea", "BitEA Register Allocator", false,
                    false)

FunctionPass *llvm::createBitEARegisterAllocator() {
  return new RABitEA();
}

FunctionPass *llvm::createBitEARegisterAllocator(RegAllocFilterFunc F) {
  return new RABitEA(F);
}


bool RABitEA::LRE_CanEraseVirtReg(Register VirtReg) {
  LiveInterval &LI = LIS->getInterval(VirtReg);
  if (VRM->hasPhys(VirtReg)) {
    Matrix->unassign(LI);
    aboutToRemoveInterval(LI);
    return true;
  }
  // Unassigned virtreg is probably in the priority queue.
  // RegAllocBase will erase it after dequeueing.
  // Nonetheless, clear the live-range so that the debug
  // dump will show the right state for that VirtReg.
  LI.clear();
  return false;
}

void RABitEA::LRE_WillShrinkVirtReg(Register VirtReg) {
  if (!VRM->hasPhys(VirtReg))
    return;

  // Register is assigned, put it back on the queue for reassignment.
  LiveInterval &LI = LIS->getInterval(VirtReg);
  Matrix->unassign(LI);
  enqueue(&LI);
}

RABitEA::RABitEA(RegAllocFilterFunc F)
    : MachineFunctionPass(ID), RegAllocBase(F) {}

void RABitEA::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addPreserved<LiveIntervalsWrapperPass>();
  AU.addPreserved<SlotIndexesWrapperPass>();
  AU.addRequired<LiveDebugVariables>();
  AU.addPreserved<LiveDebugVariables>();
  AU.addRequired<LiveStacks>();
  AU.addPreserved<LiveStacks>();
  AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
  AU.addPreserved<MachineBlockFrequencyInfoWrapperPass>();
  AU.addRequiredID(MachineDominatorsID);
  AU.addPreservedID(MachineDominatorsID);
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  AU.addRequired<VirtRegMap>();
  AU.addPreserved<VirtRegMap>();
  AU.addRequired<LiveRegMatrix>();
  AU.addPreserved<LiveRegMatrix>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RABitEA::releaseMemory() {
  SpillerInstance.reset();
}


// Spill or split all live virtual registers currently unified under PhysReg
// that interfere with VirtReg. The newly spilled or split live intervals are
// returned by appending them to SplitVRegs.
bool RABitEA::spillInterferences(const LiveInterval &VirtReg,
                                 MCRegister PhysReg,
                                 SmallVectorImpl<Register> &SplitVRegs) {
  // Record each interference and determine if all are spillable before mutating
  // either the union or live intervals.
  SmallVector<const LiveInterval *, 8> Intfs;

  // Collect interferences assigned to any alias of the physical register.
  for (MCRegUnit Unit : TRI->regunits(PhysReg)) {
    LiveIntervalUnion::Query &Q = Matrix->query(VirtReg, Unit);
    for (const auto *Intf : reverse(Q.interferingVRegs())) {
      if (!Intf->isSpillable() || Intf->weight() > VirtReg.weight())
        return false;
      Intfs.push_back(Intf);
    }
  }
  LLVM_DEBUG(dbgs() << "spilling " << printReg(PhysReg, TRI)
                    << " interferences with " << VirtReg << "\n");
  assert(!Intfs.empty() && "expected interference");

  // Spill each interfering vreg allocated to PhysReg or an alias.
  for (const LiveInterval *Spill : Intfs) {
    // Skip duplicates.
    if (!VRM->hasPhys(Spill->reg()))
      continue;

    // Deallocate the interfering vreg by removing it from the union.
    // A LiveInterval instance may not be in a union during modification!
    Matrix->unassign(*Spill);

    // Spill the extracted interval.
    LiveRangeEdit LRE(Spill, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
    spiller().spill(LRE);
  }
  return true;
}

// Driver for the register assignment and splitting heuristics.
// Manages iteration over the LiveIntervalUnions.
//
// This is a minimal implementation of register assignment and splitting that
// spills whenever we run out of registers.
//
// selectOrSplit can only be called once per live virtual register. We then do a
// single interference test for each register the correct class until we find an
// available register. So, the number of interference tests in the worst case is
// |vregs| * |machineregs|. And since the number of interference tests is
// minimal, there is no value in caching them outside the scope of
// selectOrSplit().
MCRegister RABitEA::selectOrSplit(const LiveInterval &VirtReg,
                                  SmallVectorImpl<Register> &SplitVRegs) {
  // Populate a list of physical register spill candidates.
  SmallVector<MCRegister, 8> PhysRegSpillCands;

  // Check for an available register in this class.
  auto Order =
      AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);
  for (MCRegister PhysReg : Order) {
    assert(PhysReg.isValid());
    // Check for interference in PhysReg
    switch (Matrix->checkInterference(VirtReg, PhysReg)) {
    case LiveRegMatrix::IK_Free:
      // PhysReg is available, allocate it.
      return PhysReg;

    case LiveRegMatrix::IK_VirtReg:
      // Only virtual registers in the way, we may be able to spill them.
      PhysRegSpillCands.push_back(PhysReg);
      continue;

    default:
      // RegMask or RegUnit interference.
      continue;
    }
  }

  // Try to spill another interfering reg with less spill weight.
  for (MCRegister &PhysReg : PhysRegSpillCands) {
    if (!spillInterferences(VirtReg, PhysReg, SplitVRegs))
      continue;

    assert(!Matrix->checkInterference(VirtReg, PhysReg) &&
           "Interference after spill.");
    // Tell the caller to allocate to this newly freed physical register.
    return PhysReg;
  }

  // No other spill candidates were found, so spill the current VirtReg.
  LLVM_DEBUG(dbgs() << "spilling: " << VirtReg << '\n');
  if (!VirtReg.isSpillable())
    return ~0u;
  LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  spiller().spill(LRE);

  // The live virtual register requesting allocation was spilled, so tell
  // the caller not to allocate anything during this round.
  return 0;
}

//Main function for building the Interference Graph
void RABitEA::buildInterGraph(){
  inter_graph.size = MRI->getNumVirtRegs();
  inter_graph.weights = TRI->getRegisterCosts(*MF).data();
  inter_graph.edge_mat = 
    (block_t*)calloc(inter_graph.size, TOTAL_BLOCK_NUM(inter_graph.size)*sizeof(block_t));

  block_t (*edge_mat)[][TOTAL_BLOCK_NUM(inter_graph.size)] = 
      (block_t (*)[][TOTAL_BLOCK_NUM(inter_graph.size)])inter_graph.edge_mat;

	for(int i = 0; i < inter_graph.size; ++i) {
  	unsigned Reg = Register::index2VirtReg(i);
  	if(MRI->reg_nodbg_empty(Reg))
      continue;
    LiveInterval *VirtReg = &LIS->getInterval(Reg);

    for(int j = i; j <= inter_graph.size; ++j) {
      unsigned Reg = Register::index2VirtReg(j);
      if(MRI->reg_nodbg_empty(Reg))
        continue;
      LiveInterval *VirtReg2 = &LIS->getInterval(Reg);

      if (VirtReg->overlaps(VirtReg2)) {
        SET_BIT((*edge_mat)[i], j);
        SET_BIT((*edge_mat)[j], i);
      }
    }
  }
}

bool RABitEA::runOnMachineFunction(MachineFunction &mf) {
  LLVM_DEBUG(dbgs() << "********** BitEA REGISTER ALLOCATION **********\n"
                    << "********** Function: " << mf.getName() << '\n');

  MF = &mf;
  RegAllocBase::init(getAnalysis<VirtRegMap>(),
                     getAnalysis<LiveIntervalsWrapperPass>().getLIS(),
                     getAnalysis<LiveRegMatrix>());
  VirtRegAuxInfo VRAI(
      *MF, *LIS, *VRM, getAnalysis<MachineLoopInfoWrapperPass>().getLI(),
      getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI());
  VRAI.calculateSpillWeightsAndHints();

  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM, VRAI));

  buildInterGraph();

  char buffer[128];
  sprintf(buffer, "%s.edgelist", mf.getName().data());
  write_graph(buffer, inter_graph.size, inter_graph.edge_mat, 0);

  sprintf(buffer, "%s.col.w", mf.getName().data());
  write_weights(buffer, inter_graph.size, inter_graph.weights);

  allocatePhysRegs();
  postOptimization();

  // Diagnostic output before rewriting
  LLVM_DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << *VRM << "\n");

  releaseMemory();
  return true;
}
