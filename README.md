# Hackathon ONERA - Détection de signaux radio avec GNU Radio et PyTorch
## Avancement du projet

Le projet dispose maintenant d’un premier socle fonctionnel autour de l’intégration `GNU Radio` / `PyTorch` :

- un flowgraph `GNU Radio` est disponible dans `main_gnuradio/` ;
- un bloc Python embarqué charge un modèle TorchScript et agit comme détecteur `signal/bruit` ;
- un modèle de démonstration est fourni dans `models/signal_noise_detector.pt` ;
- une version Python directe basée sur `pyrtlsdr` est disponible dans `main_pyrtlsdr/` pour tester la chaîne hors `GNU Radio` ;
- un second modèle, orienté détection IQ plus robuste, est préparé via `models/classifier_signal.py` et exporté sous `signal_detector_iq.pt`.

La suite du travail consiste surtout à fiabiliser l’entrée des buffers IQ, améliorer le modèle de détection, et documenter proprement l’installation et l’exécution des différents prototypes.

## Installation

### Avec conda (recommandé — inclut GNU Radio)

```bash
conda env create -f environment.yml
conda activate hackathon-onera
```

### Avec pip seul (sans GNU Radio)

> GNU Radio doit être installé séparément via conda pour que les blocs GNU Radio utilisant pytorch fonctionnent correctement.

```bash
pip install -r requirements.txt
```

### Problèmes d'antenne avec Pyrtlsdr

> La solution ci-dessous a été trouvée après de nombreuses recherches infructueuses en ligne, suite à une demande à un agent IA.

```bash
echo "1-1:1.0" | sudo tee /sys/bus/usb/drivers/dvb_usb_rtl28xxu/unbind
sudo modprobe -r rtl2832_sdr rtl2832 dvb_usb_rtl28xxu dvb_usb_v2
lsmod | grep -E 'rtl2832|dvb_usb_rtl28xxu|dvb_usb_v2' || true

printf "blacklist dvb_usb_rtl28xxu\nblacklist dvb_usb_v2\nblacklist rtl2832\nblacklist rtl2832_sdr\n" | sudo tee /etc/modprobe.d/blacklist-rtl-sdr-dvb.conf
sudo update-initramfs -u
```

## Utilisation rapide

### 1) Détection directe RTL-SDR

```bash
python main_pyrtlsdr/main.py
```

Ce script lit quelques buffers IQ sur un dongle RTL-SDR, charge `models/signal_noise_detector.pt` et affiche la classe prédite.

### 2) Prototype GNU Radio

Ouvrez `main_gnuradio/main.grc` dans GNU Radio Companion, puis lancez le flowgraph généré.  
Le bloc Python embarqué charge aussi `models/signal_noise_detector.pt`.

### 3) Génération / entraînement de modèles

- `models/create_empty_model.py` crée un modèle TorchScript minimal ;
- `models/classifier_signal.py` entraîne un détecteur IQ simple puis exporte `signal_detector_iq.pt`.

## Prototypes et scripts utiles

- `main_gnuradio/main.py` : flowgraph de démonstration avec bloc Python embarqué ;
- `main_pyrtlsdr/main.py` : lecture RTL-SDR et inférence PyTorch directe ;
- `models/create_empty_model.py` : génération du modèle TorchScript placeholder ;
- `models/classifier_signal.py` : entraînement / export d’un détecteur IQ plus complet.

## Données

Le dépôt ne contient pas de jeu de données externe. Les scripts s’appuient soit sur des signaux synthétiques générés localement, soit sur un flux RTL-SDR en direct. Si vous utilisez des captures externes, il faut les transmettre séparément.