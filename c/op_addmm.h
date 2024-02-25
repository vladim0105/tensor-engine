#pragma once
#include "engine.h"
// x+y*z
typedef struct {
  te_tensor_t x;
  te_tensor_t y;
  te_tensor_t z;
  te_tensor_t out;
} op_addmm_args_t;

size_t op_addmm_parse_args(
    te_program_t* program,
    uint32_t instruction_offset,
    void* out);

bool op_addmm_execute(te_program_t* program, op_addmm_args_t* args);