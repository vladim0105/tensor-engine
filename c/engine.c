#include "engine.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "op_addmm.h"
#include "op_permute.h"

bool execute(te_program_t* program) {
  for (size_t i = 0; i < program->instruction_size; i++) {
    te_opcode_t opcode = program->instructions[i];

    switch (opcode) {
      case TE_OP_PERMUTE: {
        op_permute_args_t args;
        i += op_permute_parse_args(program, i + 1, &args);
        op_permute_execute(program, &args);
      } break;
      case TE_OP_ADDMM: {
        op_addmm_args_t args;
        i += op_addmm_parse_args(program, i + 1, &args);
        op_addmm_execute(program, &args);
      } break;
      default:
        break;
    }
  }
}

size_t parse_tensor(
    te_program_t* program,
    uint32_t instruction_offset,
    te_tensor_t* out) {
  const uint8_t* instruction = &program->instructions[instruction_offset];
  size_t offset = 0;

  out->ndim = &instruction[0];
  offset += sizeof(uint8_t);

  out->shape = &instruction[1];
  offset += *out->ndim * sizeof(uint8_t);

  te_data_location_t data_location = (te_data_location_t)instruction[offset];
  offset += sizeof(uint8_t);

  uint32_t data_offset = instruction[offset];
  offset += sizeof(uint32_t);

  switch (data_location) {
    case TE_DATA_LOCATION_ARENA:
      out->data = &program->arena_memory[data_offset];
      break;
    case TE_DATA_LOCATION_CONSTANT:
      out->data = &program->constant_memory[data_offset];
      break;
    default:
      break;
  }

  out->numel = 1;
  for (size_t i = 0; i < *out->ndim; i++) {
    out->numel *= out->shape[i];
  }

  return offset;
}