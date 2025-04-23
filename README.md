# Manuel image_clustering_cadastre


### Description

Outil pour clusteriser des images en se basant sur leurs histogrammes.

### Installation

1. **Prérequis**
   - Python
   - Git

2. **Configuration de l'environnement**
   ```bash
   # Cloner le dépôt
   git clone https://github.com/oeemor/image_clustering_cadastre.git
   cd image_clustering_cadastre

   # Créer et activer un environnement virtuel
   python3 -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate

   # Installer les dépendances
   pip install -r requirements.txt
   ```


### Fonctionnement

Lancez l'invite de commande :
```bash
python histo.py --dir /chemin/vers/les/images --nb-clusters <k> --show (optionnel)
```
