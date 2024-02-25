#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint8_t* arena_memory;
  const uint32_t arena_size;
  const uint8_t* constant_memory;
  const uint32_t constant_size;
  const uint8_t* instructions;
  const uint32_t instruction_size;
  uint32_t input_offset;
  uint32_t output_offset;
} te_program_t;

typedef struct {
  void* data;
  uint32_t* offset;
  const uint8_t* ndim;
  const uint8_t* shape;
  uint32_t numel;
} te_tensor_t;

typedef enum {
  TE_OP_ADD = 0x00,
  TE_OP_MUL = 0x01,
  TE_OP_PERMUTE = 0x02,
  TE_OP_ADDMM = 0x03,
} te_opcode_t;

typedef enum {
  TE_DATA_LOCATION_ARENA = 0x00,
  TE_DATA_LOCATION_CONSTANT = 0x01,
} te_data_location_t;

bool execute(te_program_t* program);
size_t parse_tensor(
    te_program_t* program,
    uint32_t instruction_offset,
    te_tensor_t* out);