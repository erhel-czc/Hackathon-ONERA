### Nature du livrable

GNUradio est un logiciel en libre-accès utilisé pour faire du traitement de signal radio. Il prend la forme d'une programmation par bloc. Les blocs sont codés en python ou C++ et il est possible d'ajouter des blocs python à ceux déjà développés. L'objectif de notre travail était de proposer un placeholder pouvant accueillir un modèle de réseau de neurones. Pour cela, nous proposons un fichier GNU avec un floxgraph possible et nous avons mis un réseau trivial non entraîné. 

### Valeur ajoutée

Notre travail a consisté à développer un bloc python pour gnu radio capable d'accueillir un réseau de neurones pour l'identification d'un signal au sein d'un bruit.

### Difficultés rencontrées

Ensuite, les principaux problèmes que nous avons rencontré concernent l'utilisation de gnuradio et l'import de programme python. En effet, l'import de pytorch a été compliqué et il a été nécessaire de réaliser l'installation de gnuradio à plusieurs reprises en adaptant l'environnement dans lequel le logiciel est installé. 

Par ailleurs, les réseaux de neurones doivent impérativement prendre en entrée un nombre fixé de données (puisque le nombre de neurones est considéré comme un hyperparamètre). Or, les blocs simulant un signal sur gnou peuvent produire des signaux de tailles très variées :  en sortie des blocs on trouve des listes de complexes de taille variant entre quelques dizaines et 4096. Afin d'identifier, nous avons dû créer dans gnu un bloc de débuggage (affichage des longueurs des listes). Puis pour résoudre ce problème, il a été nécessaire de produire un filtre qui ne laisse passer que les listes de nombre suffisamment longues (ou du moins de longueur compatible avec notree domaine).

Nous avons en outre eu des difficultés à charger des modèles entraînés dans gnu. Il a fallu gérer intelligemment les chemins d'accès aux modèles afin que gnu puisse les retrouver.

### Licence

This project is licensed under the terms of the MIT license.
