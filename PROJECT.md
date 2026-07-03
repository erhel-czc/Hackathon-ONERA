### Valeur ajoutée

Notre travail a consisté à développer un bloc python pour gnu radio capable d'accueillir un réseau de neurones pour l'identification d'un signal au sein d'un bruit.

### Difficultés rencontrées

Ensuite, les principaux problèmes que nous avons rencontré concernent l'utilisation de gnuradio et l'import de programme python. En effet, l'import de pytorch a été compliqué et il a été nécessaire de réaliser l'installation de gnuradio à plusieurs reprises en adaptant l'environnement dans lequel le logiciel est installé. 

Par ailleurs, les réseaux de neurones doivent impérativement prendre en entrée un nombre fixé de données (puisque le nombre de neurones est considéré comme un hyperparamètre). Or, les blocs simulant un signal sur gnou peuvent produire des signaux de tailles très variées :  en sortie des blocs on trouve des listes de complexes de taille variant entre quelques dizaines et 4096. Afin d'identifier, nous avons dû créer dans gnu un bloc de débuggage (affichage des longueurs des listes). Nous avons ensuite créer un bloc qui fonctionnait comme une file d'attente et ne renvoyant que des listes de complexe de taille constante. Cependant il semble que GNU Radio, ne tient pas à maintenir l'unité de ces listes, qui se trouvait donc décomposé au bloc d'après. Avec plus de temps on aurait pu essayer de passer de Stream en Vector. Cependant on a trouvé une solution plus simple, mettre dans un même bloc ce filtre, et le réseau de neuronnes.

Nous avons en outre eu des difficultés à charger des modèles entraînés dans gnu. Il a fallu gérer intelligemment les chemins d'accès aux modèles afin que gnu puisse les retrouver.

### Licence

This project is licensed under the terms of the MIT license.
