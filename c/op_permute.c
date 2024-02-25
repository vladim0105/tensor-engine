#include "op_permute.h"
#include <stddef.h>
#include <stdio.h>
size_t op_permute_parse_args(
    te_program_t* program,
    uint32_t instruction_offset,
    void* out) {
  op_permute_args_t* args = (op_permute_args_t*)out;
  size_t offset = 0;

  offset += parse_tensor(program, instruction_offset + offset, &args->x);
  offset +=
      parse_tensor(program, instruction_offset + offset, &args->permutation);
  offset += parse_tensor(program, instruction_offset + offset, &args->out);

  return offset;
}

bool op_permute_execute(te_program_t* program, op_permute_args_t* args) {
  // Given a permutation array, permute the input tensor
  // Example [1, 0] permutation will swap the first and second dimensions

  for (size_t i = 0; i < args->out.numel; i++) {
    float* out_data = (float*)args->out.data;
    float* in_data = (float*)args->x.data;
    *out_data = *in_data;
  }

  return true;
}