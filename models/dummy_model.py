import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_features=in_chs, out_features=out_chs))

    def forward(self, x):
        return self.model(x)

# model = DummyModel(4092, 10)
# model.eval()

# example_input = torch.randn(1, 4092)
# scripted_model = torch.jit.trace(model, example_input)

# PATH = "dummymodel.pt"
# scripted_model.save(PATH)

m = torch.jit.load("dummymodel.pt")
print(m)