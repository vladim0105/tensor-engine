#pragma once
#include "engine.h"

typedef struct {
  te_tensor_t x;
  te_tensor_t permutation;
  te_tensor_t out;
} op_permute_args_t;

size_t op_permute_parse_args(
    te_program_t* program,
    uint32_t instruction_offset,
    void* out);

bool op_permute_execute(te_program_t* program, op_permute_args_t* args);
