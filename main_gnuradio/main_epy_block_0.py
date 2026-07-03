import numpy as np
import os
import torch
from gnuradio import gr


class blk(gr.sync_block):
    """Model-based signal/noise detector and gate for complex IQ streams."""

    def __init__(self,
                 model_path='/home/erhelito/Nextcloud/files/travail/info/Hackathon-ONERA/models/signal_noise_detector.pt',
                 threshold=0.5):
        gr.sync_block.__init__(
            self,
            name='Signal/Noise Detector Block',
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        print('current working directory:', os.getcwd())

        self.threshold = float(threshold)
        self.model_path = model_path
        self.model = self.load_model(
            model_path)
        self.last_probability = 0.0
        self.last_class_name = 'noise'
        self._warned_once = False

    def load_model(self, model_path):
        model = torch.jit.load(model_path)
        model.eval()

        return model

    def predict_proba(self, model, x):
        with torch.no_grad():
            out = model(x)

            return float(out.reshape(-1)[0].item())

    def predict_class(self, model, x, threshold=0.5):
        probability = self.predict_proba(model, x)
        class_id = int(probability >= threshold)
        class_name = 'signal' if class_id == 1 else 'noise'

        return class_id, class_name, probability

    def work(self, input_items, output_items):
        iq = input_items[0]
        # Expected model input: batch of IQ samples encoded as [real, imag].
        features = np.stack((iq.real, iq.imag), axis=-
                            1).astype(np.float32, copy=False)
        x = torch.from_numpy(features).unsqueeze(0)

        try:
            class_id, class_name, probability = self.predict_class(self.model,
                                                                   x,
                                                                   threshold=self.threshold)
            self.last_probability = probability
            self.last_class_name = class_name

            if class_id == 1:
                output_items[0][:] = iq
            else:
                output_items[0][:] = np.complex64(0.0 + 0.0j)

        except Exception as exc:
            # Keep the flowgraph alive and bypass data if inference fails.
            if not self._warned_once:
                print('Signal/Noise Detector Block inference error:', exc)
                self._warned_once = True

            output_items[0][:] = iq

        return len(output_items[0])
