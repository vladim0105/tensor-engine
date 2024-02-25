import torch
from torch.export import export
from torch._export import capture_pre_autograd_graph
import executorch.exir as exir
import export as test
import passes
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
        self.linear_out = torch.nn.Linear(5, 1)
        self.gru = torch.nn.GRU(5, 5)

    def forward(self, x):
        x = self.linear(x)
        # h0 = torch.zeros(1, 5)
        # x, _ = self.gru(x, h0)
        x = self.linear_out(x)
        return x
    
m = SimpleModel()
example_args = (torch.randn(1,5),)

a = export(m, example_args)
a = passes.extract_constants(a)
print(a)

test.export(a)

