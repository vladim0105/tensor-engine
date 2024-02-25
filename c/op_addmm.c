#include "op_addmm.h"
#include <stdio.h>
size_t op_addmm_parse_args(
    te_program_t* program,
    uint32_t instruction_offset,
    void* out) {
  op_addmm_args_t* args = (op_addmm_args_t*)out;
  size_t offset = 0;

  offset += parse_tensor(program, instruction_offset + offset, &args->x);
  offset += parse_tensor(program, instruction_offset + offset, &args->y);
  offset += parse_tensor(program, instruction_offset + offset, &args->z);
  offset += parse_tensor(program, instruction_offset + offset, &args->out);

  return offset;
}

bool op_addmm_execute(te_program_t* program, op_addmm_args_t* args) {
  // Check if the dimensions are compatible
  if (args->y.shape[1] != args->z.shape[0]) {
    return false;
  }
  // Do the matrix multiply
  for (size_t i = 0; i < args->out.shape[0]; i++) {
    for (size_t j = 0; j < args->out.shape[1]; j++) {
      float* out_data = (float*)args->out.data + i * args->out.shape[1] + j;
      *out_data = 0;
      for (size_t k = 0; k < args->y.shape[1]; k++) {
        float* y_data = (float*)args->y.data + i * args->y.shape[1] + k;
        float* z_data = (float*)args->z.data + k * args->z.shape[1] + j;
        *out_data += *y_data * *z_data;
        // Add the x tensor
        float* x_data = (float*)args->x.data + i * args->x.shape[1] + j;
        *out_data += *x_data;
      }
    }
  }
}