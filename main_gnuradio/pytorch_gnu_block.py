import numpy as np
from gnuradio import gr
import torch


class blk(gr.sync_block):
    """
    GNU Radio embedded Python block performing windowed binary detection
    via a PyTorch model.

    The GNU Radio scheduler delivers input buffers of variable size. This
    block accumulates incoming complex64 samples into a fixed-size window,
    runs inference once the window is full, and outputs the detection score
    (post-sigmoid, in [0, 1]) replicated over the window duration as a
    float32 stream.

    Signal representation
    ---------------------
    A window of N complex64 samples is decomposed into a concatenated
    vector of shape (1, 2N):

        tensor_in = [re_0, re_1, ..., re_{N-1}, im_0, im_1, ..., im_{N-1}]

    The model returns a single logit per window. This block applies a
    sigmoid and writes the resulting scalar, replicated N times, to the
    output stream.

    Parameters
    ----------
    model_path : str
        Absolute path to the PyTorch model file (.pt), exported via
        torch.jit.trace() or torch.jit.script() and saved with .save().
        Model must accept input of shape (1, 2 * window_size) and return
        a single logit of shape (1,) or (1, 1).
    device : str
        PyTorch device string ('cpu' or 'cuda:0'). Defaults to 'cpu'.
    window_size : int
        Number of complex64 samples per inference window. Defaults to 4092.
    threshold : float
        If >= 0.0, output is hard-thresholded (0.0 or 1.0) at this value
        applied to the post-sigmoid score. If < 0.0, the continuous score
        in [0, 1] is output directly. Defaults to -1.0 (no thresholding).

    Attributes
    ----------
    in_buffer : np.ndarray, shape (window_size,), dtype=complex64
        Circular accumulation buffer for incoming samples.
    in_buffer_idx : int
        Current write index into in_buffer. Resets to 0 after each inference.
    out_buffer : np.ndarray, dtype=float32
        Dynamic output buffer accumulating replicated decision values,
        consumed progressively at each work() call.

    WARNING
    -------
    the model_path default arg in the python file must be a valid absolute
    file path on your computer before using GNU !
    """

    def __init__(self, model_path="C:\\Users\\multi\\Desktop\\Hackathon\\dummymodel.pt",
                 device="cpu", window_size=4092, threshold=-1.0):
        gr.sync_block.__init__(
            self,
            name='PyTorch Detection Block',
            in_sig=[np.complex64],
            out_sig=[np.float32]
        )

        self.device = device
        self.window_size = window_size
        self.threshold = threshold
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        self.in_buffer = np.zeros(window_size, dtype=np.complex64)
        self.in_buffer_idx = 0

        self.out_buffer = np.zeros(0, dtype=np.float32)

    def run_inference(self):
        """
        Execute one forward pass on the current input window.

        Returns
        -------
        np.ndarray, shape (window_size,), dtype=float32
            Detection score (or hard decision if threshold >= 0.0),
            replicated over the window length.
        """
        real = torch.from_numpy(self.in_buffer.real.copy()).float()
        imag = torch.from_numpy(self.in_buffer.imag.copy()).float()
        tensor_in = torch.cat([real, imag], dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(tensor_in).squeeze()
            score = torch.sigmoid(logit).item()

        if self.threshold >= 0.0:
            score = 1.0 if score >= self.threshold else 0.0

        return np.full(self.window_size, score, dtype=np.float32)

    def work(self, input_items, output_items):
        """
        GNU Radio work() callback — called by the scheduler on each buffer cycle.

        Accumulates input samples into in_buffer. Each time in_buffer reaches
        window_size samples, run_inference() is called and its output is
        appended to out_buffer. The output array is then filled from
        out_buffer.

        Parameters
        ----------
        input_items : list of np.ndarray
            input_items[0] : complex64 array of N incoming samples, N variable.
        output_items : list of np.ndarray
            output_items[0] : float32 array of N samples to write.

        Returns
        -------
        int
            Number of output items produced (always N).
        """
        in0 = input_items[0]
        out = output_items[0]
        N = len(in0)

        src_idx = 0
        remaining = N

        while remaining > 0:
            space = self.window_size - self.in_buffer_idx
            to_copy = min(space, remaining)

            self.in_buffer[self.in_buffer_idx:self.in_buffer_idx + to_copy] = \
                in0[src_idx:src_idx + to_copy]

            self.in_buffer_idx += to_copy
            src_idx += to_copy
            remaining -= to_copy

            if self.in_buffer_idx == self.window_size:
                processed = self.run_inference()
                self.out_buffer = np.concatenate([self.out_buffer, processed])
                self.in_buffer_idx = 0

        if len(self.out_buffer) >= N:
            out[:] = self.out_buffer[:N]
            self.out_buffer = self.out_buffer[N:]
        else:
            out[:] = 0.0

        return N