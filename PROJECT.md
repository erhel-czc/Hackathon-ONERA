### Nature du livrable

GNU Radio est un environnement open source de traitement du signal par blocs.  
Le projet fournit une intégration **placeholder** d’un réseau de neurones dans cette chaîne, avec :
- un flowgraph GNU Radio (`main_gnuradio/main.grc`) ;
- un bloc Python embarqué de détection (`main_gnuradio/main_epy_block_0.py`) ;
- des scripts de création/chargement de modèles TorchScript (`models/`) ;
- une variante hors GNU Radio basée sur `pyrtlsdr`, développée sur les derniers jours (`main_pyrtlsdr/main.py`).

L’objectif atteint est la démonstration d’une intégration de modèle PyTorch dans un traitement IQ, plus qu’une performance de détection optimisée.

### Démarche et choix techniques

Nous avons commencé par un modèle Torch très simple pour valider l’intégration de bout en bout, puis nous avons gardé la même interface pour un second modèle plus réaliste basé sur des fenêtres IQ de taille fixe.

Le dépôt est organisé en trois briques :
- `main_gnuradio/` pour l’intégration temps réel dans GNU Radio ;
- `main_pyrtlsdr/` pour tester la chaîne sans flowgraph ;
- `models/` pour l’entraînement et l’export des modèles.

### Valeur ajoutée

La contribution principale est d’avoir relié un flux IQ complexe à une inférence réseau de neurones dans un bloc GNU Radio, puis d’avoir documenté une architecture réutilisable pour brancher un modèle entraîné ultérieurement.

Le dépôt contient ainsi une base fonctionnelle pour :
- injecter un modèle TorchScript (`.pt`) ;
- traiter un flux IQ en temps réel ;
- décider « signal / bruit » et propager le résultat dans la chaîne (cette partie nécessite un entrainement du modèle sur un jeu de données que nous n'avions pas, actuellement le modèle renvoie de fausses valeurs)

### Difficultés rencontrées

Les principaux points bloquants ont été :
- la configuration conjointe de GNU Radio et PyTorch selon l’environnement d’exécution ;
- la gestion des tailles variables de buffers IQ côté GNU Radio, alors que les réseaux de neurones attendent une taille d’entrée fixe ;
- le chargement fiable des modèles (chemins de fichiers, exécution depuis différents répertoires).
- fonctionnement de l'antenne radio RTL-SDR sur certaines machines.

Une partie du travail a consisté à instrumenter les blocs pour observer les tailles réellement reçues, puis à adapter la logique de préparation des données avant inférence.
Par ailleurs, les réseaux de neurones doivent impérativement prendre en entrée un nombre fixe de données (puisque le nombre de neurones est considéré comme un hyperparamètre). Or, les blocs simulant un signal sur GNU peuvent produire des signaux de tailles très variées : en sortie des blocs, on trouve des listes de complexes dont la taille varie entre 1 et 4096 IQ. Afin d'identifier le problème, nous avons dû créer dans GNU un bloc de débogage (affichage des longueurs des listes). Nous avons ensuite créé un bloc qui fonctionnait comme une file d'attente et ne renvoyait que des listes de complexes de taille constante. Néanmoins, il semble que GNU Radio ne tienne pas à maintenir l'unité de ces listes, qui se trouvaient donc décomposées dans le bloc suivant. Avec plus de temps, nous aurions pu essayer de passer de Stream à Vector. Cependant, nous avons trouvé une solution plus simple : mettre dans un même bloc ce filtre et le réseau de neurones.

De la même manière, pour créer une base de données d'entraînement, nous avons eu plus ou moins le même problème de gestion des tailles de buffers, d'autant plus qu'une seconde de données correspondait à plus de 1 Go de données. Nous avons donc placé un bloc de filtrage qui renvoyait une séquence de taille 4096 avant le bloc d'enregistrement dans le fichier de données.

Concernant la gestion des chemins de modèles, nous avons été forcés de fixer le chemin en absolu, car gnuradio exécute le flowgraph depuis un répertoire temporaire avant de lancer le script principal dans le répertoire du projet, ce qui rendait les chemins relatifs non fiables.

Nous avons également rencontré des problèmes de fonctionnement de l'antenne radio RTL-SDR sur certaines machines, celle-ci ne fonctionnant que sur l'unique ordinateur sous Linux du groupe. Pour pouvoir utiliser l'antenne avec `pyrtlsdr`, nous avons dû désactiver le module du noyau qui interférait avec le fonctionnement de l'antenne (procédure documentée dans le fichier `README.md`).

### Fonctionnement et limites

Le code est exécutable avec les dépendances indiquées dans `requirements.txt` et/ou `environment.yml`, mais GNU Radio doit être installé via conda.

### Licence

Ce projet est distribué sous licence MIT (voir `LICENSE`).
