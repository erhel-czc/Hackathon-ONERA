# main_pyrtlsdr

Adaptation de `main_gnuradio/` vers une version Python directe basée sur `pyrtlsdr`.

## Dépendances

- `pyrtlsdr`
- `numpy`
- `matplotlib`
- `torch`

## Utilisation

Depuis la racine du projet :

```bash
python3 main_pyrtlsdr/main.py
```

Les réglages (fréquence, gain, modèle, etc.) sont définis directement en constantes en haut de `main_pyrtlsdr/main.py`.

## Ce que fait le programme

- Lit des échantillons IQ depuis un récepteur RTL-SDR (`pyrtlsdr`).
- Applique le modèle PyTorch `signal_noise_detector.pt`.
- Affiche en temps réel :
  - IQ temporel (I/Q),
  - Constellation de la sortie détectée,
  - Waterfall du signal brut reçu.
