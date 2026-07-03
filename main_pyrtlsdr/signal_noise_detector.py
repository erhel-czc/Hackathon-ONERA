from pathlib import Path

import numpy as np
import torch


class SignalNoiseDetector:
    """PyTorch-based signal/noise detector for IQ samples."""

    def __init__(self, model_path, threshold=0.5):
        self.threshold = float(threshold)
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = torch.jit.load(str(self.model_path), map_location="cpu")
        self.model.eval()
        self.expected_input_size = self._infer_expected_input_size()

    def _infer_expected_input_size(self):
        """Infer the expected flattened input size from a first linear layer."""
        for param in self.model.parameters():
            if param.ndim == 2:
                in_features = int(param.shape[1])
                if in_features > 4:
                    return in_features
        return None

    def _fit_vector(self, vector, target_size):
        """Pad or truncate a 1D float32 vector to a fixed size."""
        size = int(vector.size)
        if size == target_size:
            return vector
        if size > target_size:
            return vector[:target_size]
        out = np.zeros(target_size, dtype=np.float32)
        out[:size] = vector
        return out

    def predict_class(self, iq):
        flat_ri = np.concatenate((iq.real, iq.imag)).astype(
            np.float32, copy=False)
        if self.expected_input_size is not None:
            flat_ri = self._fit_vector(flat_ri, self.expected_input_size)

        x = torch.from_numpy(flat_ri).unsqueeze(0)

        with torch.no_grad():
            out = self.model(x)
            probability = float(out.reshape(-1)[0].item())

        class_id = int(probability >= self.threshold)
        class_name = "signal" if class_id == 1 else "noise"
        return class_id, class_name, probability
