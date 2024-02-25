import torch.fx
from torch.export import ExportedProgram
from torch._subclasses import FakeTensorMode
from typing import Any, Tuple, NamedTuple, Optional, Dict
class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a PyTorch program.

    # General Tensor metadata
    shape : torch.Size
    dtype : torch.dtype
    requires_grad : bool
    stride : Tuple[int, ...]
    memory_format : Optional[torch.memory_format]

    # Quantization metadata
    is_quantized : bool
    qparams: Dict[str, Any]

def extract_constants(program: ExportedProgram) -> ExportedProgram:
    """
    Extracts constant arguments and pushes them outside of the graph and into the parameters/buffers of the graph module.
    Example:
    conv2d(x, weight, bias, dilation=[1, 1], padding=[1, 1], stride=[1, 1]) -> conv2d(x, dilation=args_dilation, padding=args_padding, stride=args_stride)
    """

    for node in program.graph.nodes:
        if node.op == "call_function":
            for i, arg in enumerate(node.args):
                if isinstance(arg, list):
                    extracted_arg = create_buffer(program, f"{node.name}_{i}", torch.Tensor(arg).int())
                    node.update_arg(i, extracted_arg)
            
    program.graph_module.recompile()
    program.graph.lint()

    return program

def create_buffer(program: ExportedProgram, name: str, value: torch.Tensor) -> torch.fx.Node:
    buffer_name = f"buffer_{name}"
    arg_name = f"args_{name}"

    # Register buffer and update the program state dict
    program.graph_module.register_buffer(buffer_name, value)
    program.state_dict[buffer_name] = value

    program.graph_signature.buffers = program.graph_signature.buffers + [buffer_name]
    program.graph_signature.inputs_to_buffers[arg_name] = buffer_name

    
    # Insert the placeholder at the root of the graph
    with program.graph.inserting_after(program.graph._root):
        extracted_arg = program.graph.placeholder(arg_name)

    # Give it some metadata :3
    extracted_arg.meta["val"] = FakeTensorMode().from_tensor(value)
    extracted_arg.meta["tensor_meta"] = TensorMetadata(
        shape=value.shape,
        dtype=value.dtype,
        requires_grad=value.requires_grad,
        stride=value.stride(),
        memory_format=torch.contiguous_format if value.is_contiguous() else None,
        is_quantized=False,
        qparams={}
    )

    return extracted_arg