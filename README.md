---
authors:
- affiliation: ONERA
  email: martin.bauw@onera.fr
  name: Martin Bauw
- affiliation: ONERA
  email: anthony.torre@onera.fr
  name: Anthony Torre
label: label-onera-martin-bauw
short_title: 02 ONERA - analyse de signal radio
subject: académique
title: 02 Analyse de signal radio par intégration d’un réseau de neurones dans un
  schéma GNU Radio
---


## L'entreprise

L’ONERA (Office national d'études et de recherches aérospatiales) est le principal centre de recherche français du secteur aéronautique, spatial et défense.

## Les porteurs du projet

Martin Bauw et Anthony Torre sont ingénieurs de recherche au `DEMR` (Département Électromagnétisme et Radar) au sein de l'Unité `MATS` (Méthodes Avancées en Traitement du Signal) du centre `ONERA` de Palaiseau.

## Le contexte

L'unité `MATS` du département `DEMR` est en charge du développement de méthodes de traitement pour les différents moyens expérimentaux du département. Ces traitements sont dédiés aux signaux émis et reçus par des capteurs électromagnétiques (EM).

Le spectre électromagnétique est porteur de nombreux signaux d’origine humaine. La quantité et la diversité de ces signaux et des contextes EM encouragent à évaluer des méthodes d’apprentissage statistique pour encoder et discriminer les signaux.

Cette unité souhaite étudier des méthodes s’appuyant exclusivement ou partiellement sur des réseaux de neurones pour détecter, représenter et séparer des signaux électromagnétiques quelconques. Bien que pensées pour des signaux radar, les méthodes seront transposables dans de nombreux domaines (e.g. acoustique).

## Le projet

```{image} image1.png
```

L’objectif principal est l’intégration d’un réseau de neurones dans un schéma `GNU Radio` [3] et la documentation de cette intégration.

### GNU Radio

<https://github.com/gnuradio/gnuradio> *GNU Radio is a free & open-source signal processing runtime and signal processing software development toolkit. Originally developed for use with software-defined radios and for simulating wireless communications, it's robust capabilities have led to adoption in hobbyist, academic, and commercial environments. GNU Radio has found use in software-defined radio, digital communications, nuclear physics, high- energy particle physics, astrophysics, radio astronomy and more!*

### Objectifs du projet

Le projet proposé aux élèves vise à intégrer un réseau de neurones implémenté avec `PyTorch` dans un schéma `GNU Radio` qui serait éventuellement branché sur une antenne. Ce projet est ambitieux, il  se limiterait dans un premier temps à&nbsp;: comprendre comment faire, à le faire avec un réseau de neurones quelconque *"placeholder"* et à documenter la démarche.

Cette volonté d'intégrer un réseau de neurones dans un schéma GNU Radio est inspirée par l'article [*«TorchSig: A GNU Radio Block and New Spectrogram Tools for Augmenting ML Training»* de Phil Vallance et al [3]](#label-ref-onera3)

Cette démarche pourrait mener à l'implémentation d'un bloc GNU Radio sous Python, voir  
<https://wiki.gnuradio.org/index.php?title=Creating_Python_OOT_with_gr-modtool>.

Il s'agira donc, pour les élèves, aussi bien de comprendre comment insérer quelque chose dans `GNU Radio` que de développer du code en Python.

Travailler sur `GNU Radio` est particulièrement enrichissant pour un étudiant car il s'agit d'un logiciel reconnu, plusieurs fois soutenu par le *Google Summer of Code*. Open source, `GNU Radio` est notamment régulièrement utilisé dans le cadre de projets mis en avant à la conférence *FOSDEM*, référence du logiciel libre, et pourrait ouvrir la porte du monde du logiciel libre à des étudiants motivés.

Ce projet est une opportunité de s’initier à l’emploi de réseaux de neurones pour détecter et discriminer des signaux [[1]](#label-ref-onera1), en considérant notamment les réseaux de neurones à valeurs complexes (`CVNN`) [[2]](#label-ref-onera2).

### Objectifs secondaires du projet

Pour aller plus loin, vous pourrez vous intéresser à plusieurs choses:</br>
1. La recherche d’une architecture et d’hyperparamètres optimaux pour le réseau de neurones
2. Le projet s’appuie sur des signaux simulés par `GNU Radio`, mais il pourrait déboucher sur des expériences où le signal est réel et
provient d’un capteur de type `USRP` ou `RTL-SDR` déjà mis en œuvre par le *DEMR*.

### Bibliographie

(label-ref-onera1)=
**[1]** O’SHEA, Timothy James, ROY, Tamoghna, et CLANCY, T. Charles. *« Over-the-air deep learning based radio signal classification.»* IEEE Journal of Selected Topics in Signal Processing, 2018, vol. 12, no 1, p. 168-179 <https://www.researchgate.net/publication/321794394_Over_the_Air_Deep_Learning_Based_Radio_Signal_Classification>
([ou autre lien](https://ieeexplore.ieee.org/abstract/document/8267032))

(label-ref-onera2)=
**[2]** TRABELSI, Chiheb, BILANIUK, Olexa, ZHANG, Ying, et al. *«Deep Complex Networks»*. International
Conference on Learning Representations. 2018 <https://openreview.net/pdf?id=H1T2hmZAb> ([ou autre lien](https://arxiv.org/abs/1705.09792))

(label-ref-onera3)=
**[3]** VALLANCE, Phil, OH, Erebus, MULLINS, Justin, et al. *« TorchSig: A GNU Radio Block and New Spectrogram Tools for Augmenting ML Training »*. Proceedings of the GNU Radio Conference, 2024 <https://pubs.gnuradio.org/index.php/grcon/article/view/147>.
