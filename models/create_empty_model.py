import torch
import torch.nn as nn
import os


# Input size = 2 * num_samples (flattened IQ vector).
# Output: sigmoid -> probability of being a signal (vs noise).

IQ_INPUT_SIZE = 2 * 2048  # 2048 at first, we'll have to adapt to the buffer's size


class SignalNoiseDetector(nn.Module):
    def __init__(self, input_size: int = IQ_INPUT_SIZE):
        super().__init__()
        self.layer = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.layer(x))


if __name__ == "__main__":
    model = SignalNoiseDetector(IQ_INPUT_SIZE)
    model.eval()

    example_input = torch.randn(1, IQ_INPUT_SIZE)
    scripted_model = torch.jit.trace(model, example_input)

    filename = "signal_noise_detector.pt"
    PATH = os.path.join(os.path.dirname(__file__), filename)
    scripted_model.save(PATH)
    print(f"Model saved to {PATH}")
    print(scripted_model)
