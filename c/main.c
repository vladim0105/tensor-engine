#include <stdio.h>
#include "engine.h"
#include "exported_program.h"

int main(int argc, char* argv[]) {
  float* in =
      (float*)&exported_program.arena_memory[exported_program.input_offset];
  float* out =
      (float*)&exported_program.arena_memory[exported_program.output_offset];

  // Fill in input tensor
  for (size_t i = 0; i < 5; i++) {
    in[i] = i;
  }

  execute(&exported_program);

  // Print output tensor
  for (size_t i = 0; i < 1; i++) {
    printf("%f\n", out[i]);
  }

  return 0;
}