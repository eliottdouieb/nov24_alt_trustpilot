Trust Pilot
==============================

# Installation et mise en place

1️. Cloner le dépôt
```bash
git clone https://github.com/eliottdouieb/nov24_alt_trustpilot
cd nov24_alt_trustpilot
```
2️. Créer un environnement virtuel (fortement recommandé)

```bash
python -m venv venv
venv\Scripts\activate
```

3️. Installer les dépendances

```bash
pip install -r requirements.txt
```

# Analyse exploratoire des avis Trustpilot

Prérequis
- Python ≥ 3.10
- (fortement recommandé) un environnement virtuel Python

## Exécution du notebook d’exploration
Lancer Jupyter :

```bash
jupyter notebook
```

Puis ouvrir le notebook d’exploration : ./notebooks/01_exploration_de_donnees.ipynb.

Étape obligatoire (à exécuter en premier)
La section suivante du notebook :
```md
# Import et chargement des données
```
doit impérativement être exécutée en priorité, car elle :
- charge les librairies,
- télécharge les ressources nécessaires (NLTK, spaCy)
- charge le jeu de données.

Aucune autre section ne fonctionnera correctement sans cette étape.

Une fois la section Import et chargement des données exécutée,
Les sections suivantes peuvent être lancées dans n’importe quel ordre.

Sections disponibles :
Kinjal
Analyse des taux de commentaires et de réponses selon la note + preprocessing NLP.

Julie
Répartition temporelle des commentaires (mois, saisons, années, tendances).

Laurine
Présence de réponse en fonction de la note et de la longueur des commentaires + analyse textuelle.

Quentin
Analyse de la longueur des commentaires vis-à-vis des notes et exploration lexicale.

# Preprocessing des données