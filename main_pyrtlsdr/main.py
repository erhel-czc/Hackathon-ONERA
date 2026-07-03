import os
import numpy as np
from rtlsdr import RtlSdr

from signal_noise_detector import SignalNoiseDetector

SAMPLE_RATE = 2_480_000
CENTER_FREQ = 90.4e6
GAIN = 10.0
SAMPLES_PER_FRAME = 4096
THRESHOLD = 0.5
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "signal_noise_detector.pt")

print("Chargement du modèle...")
detector = SignalNoiseDetector(model_path=MODEL_PATH, threshold=THRESHOLD)
print("Modèle chargé.")

print("Connexion au RTL-SDR...")
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN
print("RTL-SDR configuré.")

try:
    for i in range(5):
        iq = sdr.read_samples(SAMPLES_PER_FRAME).astype(np.complex64)
        class_id, class_name, prob = detector.predict_class(iq)
        print(
            f"[{i}] class={class_name} (id={class_id}) | p={prob:.3f} | len(iq)={len(iq)}")
        
except KeyboardInterrupt:
    print("Arrêt.")
finally:
    sdr.close()
    print("SDR fermé.")
