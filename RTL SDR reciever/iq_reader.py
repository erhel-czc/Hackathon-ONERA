import numpy as np

# Read complex64 data (interleaved floats)
samples = np.fromfile('output', dtype=np.complex64)

print(samples[0])

# faire fft du signal