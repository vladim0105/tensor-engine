
from torch.export import ExportedProgram
import torch
from torch._subclasses import FakeTensor
import torch.fx
from passes import TensorMetadata

class ArenaSimulator:
    def __init__(self):
        self.memory = 0

        self.allocations = []
        self.reusable_allocations = []
    
    def allocate(self, size):
        # Go through the reusable allocations and see if we can reuse any of them
        for i, (offset, used_size) in enumerate(self.reusable_allocations):
            if used_size >= size:
                self.allocations.append((offset, size))
                new_reusable_allocation = (offset+size, used_size-size)
                if new_reusable_allocation[1] > 0:
                    self.reusable_allocations[i] = new_reusable_allocation
                else:
                    self.reusable_allocations.pop(i)

                self.sort_reusable_allocations()
                return offset
        
        # If we can't reuse any of the reusable allocations, allocate a new one
        new_allocation = (self.memory, size)
        self.memory += size
        self.allocations.append(new_allocation)
        return new_allocation[0]

    def free(self, offset):
        # Move the allocation from the allocations list to the reusable allocations list
        for i, (offset_, size) in enumerate(self.allocations):
            if offset_ == offset:
                self.reusable_allocations.append((offset, size))
                self.sort_reusable_allocations()
                self.allocations.pop(i)
                return
        
        raise ValueError(f"Allocation at offset {offset} not found")
    
    def sort_reusable_allocations(self):
        # Sort the reusable allocations by size, to reduce fragmentation
        self.reusable_allocations.sort(key=lambda x: x[1])

op_mapping = {
    torch.ops.aten.add.Tensor: 0x00,
    torch.ops.aten.mul.Tensor: 0x01,
    torch.ops.aten.permute.default: 0x02,
    torch.ops.aten.addmm.default: 0x03,
}
DATA_ARENA = 0x00
DATA_CONSTANT = 0x01

def export(program: ExportedProgram):
    arena_memory, arena_offsets = _resolve_arena_memory(program)
    constant_memory, constant_offsets = _resolve_constant_memory(program)

    instructions = []
    for node in program.graph.nodes:
        if node.op == "call_function":
            instructions += _node_to_instruction(node, arena_offsets, constant_offsets)

    # Get input and output offset
    input_offset = arena_offsets[program.graph_signature.user_inputs[0]]
    output_offset = arena_offsets[program.graph_signature.user_outputs[0]]

    #Dump to header file
    with open("exported_program.h", "w") as f:
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n")
        f.write("#include \"engine.h\"\n")
        f.write(f"#define ARENA_SIZE {len(arena_memory)}\n")
        f.write(f"#define CONSTANT_SIZE {len(constant_memory)}\n")
        f.write(f"#define INSTRUCTION_SIZE {len(instructions)}\n")

        f.write(f"uint8_t arena_memory[ARENA_SIZE] = {{}};\n")
        f.write(f"const uint8_t constant_memory[CONSTANT_SIZE] = {{{', '.join(['0x'+hex(x)[2:].zfill(2) for x in constant_memory])}}};\n")

        f.write(f"const uint8_t instructions[] = {{{', '.join(['0x'+hex(x)[2:].zfill(2) for x in instructions])}}};\n")

        # Pack it all into a struct
        f.write("te_program_t exported_program = {\n")
        f.write("    .arena_memory = arena_memory,\n")
        f.write("    .arena_size = ARENA_SIZE,\n")
        f.write("    .constant_memory = constant_memory,\n")
        f.write("    .constant_size = CONSTANT_SIZE,\n")
        f.write("    .instructions = instructions,\n")
        f.write("    .instruction_size = INSTRUCTION_SIZE,\n")
        f.write(f"    .input_offset = {input_offset},\n")
        f.write(f"    .output_offset = {output_offset},\n")
        f.write("};\n")

def _node_to_instruction(node: torch.fx.Node, arena_offsets, constant_offsets) -> list[int]:
    
    kernel_id = -1
    try:
        kernel_id = op_mapping[node.target]
    except(KeyError):
        print(f"Op mapping not found for {node.target}")
    
    # First we add the kernel op code
    instruction = to_bytes(kernel_id, dtype=torch.uint8)

    # Then we add information about the arguments
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            # First we need to add shape information
            instruction += _meta_to_instruction(arg.meta)

            
            if arg.name in arena_offsets:
                # Add the data location
                instruction += to_bytes(DATA_ARENA, dtype=torch.uint8)
                # Then add the offset, cast it to int32 to avoid clipping
                instruction += to_bytes(arena_offsets[arg.name], dtype=torch.int32)
            elif arg.name in constant_offsets:
                instruction += to_bytes(DATA_CONSTANT, dtype=torch.uint8)
                instruction += to_bytes(constant_offsets[arg.name], dtype=torch.int32)
            else:
                raise ValueError(f"Unknown argument {arg.name}")
        else:
            print(f"Unknown argument {arg}")
        
        assert not isinstance(arg, list), "Lists args are not supported, they need to be converted to buffers"

    # Finally we need to tell it where to store the result
    instruction += _meta_to_instruction(node.meta)
    if node.name in arena_offsets:
        instruction += to_bytes(DATA_ARENA, dtype=torch.uint8)
        instruction += to_bytes(arena_offsets[node.name], dtype=torch.int32)
    elif node.name in constant_offsets:
        instruction += to_bytes(DATA_CONSTANT, dtype=torch.uint8)
        instruction += to_bytes(constant_offsets[node.name], dtype=torch.int32)

    return instruction

def _meta_to_instruction(meta: dict) -> list[int]:
    shape = list(meta["tensor_meta"].shape)
    # dtype = meta["tensor_meta"].dtype
    ndims = len(shape)

    instruction = []
    # We can get away with casting shape as uint8, having more than 255 dimensions is unlikely
    instruction += to_bytes(ndims, dtype=torch.uint8)
    instruction += to_bytes(shape, dtype=torch.uint8)
    

    return instruction

def _resolve_constant_memory(program: ExportedProgram):
    total_memory = []
    offset_tracker = {}
    for name, parameter in program.named_parameters():
        parameters_to_inputs = {v: k for k, v in program.graph_signature.inputs_to_parameters.items()}
        name = parameters_to_inputs[name]
        offset = len(total_memory)
        offset_tracker[name] = offset
        # Export weights as floats, but store them as bytes
        total_memory += to_bytes(parameter, dtype=torch.float32)

    for name, buffer in program.named_buffers():
        buffers_to_inputs = {v: k for k, v in program.graph_signature.inputs_to_buffers.items()}
        name = buffers_to_inputs[name]
        offset = len(total_memory)
        offset_tracker[name] = offset
        total_memory += to_bytes(buffer, dtype=torch.float32)

    return (total_memory, offset_tracker)

def _resolve_arena_memory(program: ExportedProgram):
    total_memory = 0
    arena = ArenaSimulator()
    num_users_tracker = {}
    offset_tracker = {}
    for node in program.graph.nodes:
        if node.op != "output":
            # Dont resolve memory for read only buffers
            if node.name in program.graph_signature.inputs_to_parameters:
                continue
            if node.name in program.graph_signature.inputs_to_buffers:
                continue


            if not node.name in num_users_tracker:
                num_users_tracker[node.name] = len(node.users.keys())

            val = node.meta["val"]
            assert isinstance(val, FakeTensor), f"Only FakeTensors are supported, got {node.target}"
            ft: FakeTensor = val
            tensor_memory_size = ft.numel()*ft.dtype.itemsize
            total_memory += tensor_memory_size
            offset = arena.allocate(tensor_memory_size)
            offset_tracker[node.name] = offset


            for input_node in node.all_input_nodes:
                if input_node.name in num_users_tracker:
                    num_users_tracker[input_node.name] -= 1
                    if num_users_tracker[input_node.name] == 0:
                        arena.free(offset_tracker[input_node.name])
                        num_users_tracker.pop(input_node.name)

    # Return a list of zeros to represent the memory (this is just to keep it consistent with the constant memory return type, it's not actually used)
    return ([0]*arena.memory, offset_tracker)
    
def to_bytes(data: list|torch.Tensor|int|float, dtype) -> list[int]:
    """
    Convert data to a list of bytes

    Args:
        data (list|torch.Tensor|int|float): The data to convert
        dtype (torch.dtype): The dtype to cast it to before converting to bytes
    """
    if isinstance(data, torch.Tensor):
        if dtype is not None:
            data = data.to(dtype)
        return data.view(dtype=torch.uint8).flatten().tolist()
    elif isinstance(data, list):
        data = torch.tensor(data, dtype=dtype)
        if dtype is not None:
            data = data.to(dtype)
        return data.view(dtype=torch.uint8).flatten().tolist()
        
    elif isinstance(data, int) or isinstance(data, float):
        return to_bytes([data], dtype=dtype)