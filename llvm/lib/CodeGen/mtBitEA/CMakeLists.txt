add_llvm_component_library(MTBITEA
  mtBitEA.c
  stdgraph.c

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CodeGen/mtBitEA
)
