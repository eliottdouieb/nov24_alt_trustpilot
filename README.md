Trust Pilot

Projet de topic modeling et de classification de commentaires Trustpilot à partir de techniques NLP classiques et contextuelles.
==============================

## Installation et mise en place

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

Prérequis
- Python ≥ 3.10 (3.11.9 recommandé)
- (fortement recommandé) un environnement virtuel Python

==============================

## Organisation du projet

Le projet est structuré autour de plusieurs notebooks Jupyter :

- 01_exploration_de_donnees.ipynb
- 02_preprocessing.ipynb
- 03_modelisations_.ipynb (plusieurs approches testées)

👉 Le notebook d’exploration est optionnel.
👉 Le notebook de preprocessing est obligatoire avant toute modélisation.

## Exécution d'un notebook
Lancer Jupyter :

```bash
jupyter notebook
```

## Analyse exploratoire des avis Trustpilot

01_exploration_de_donnees.ipynb

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

Une fois cette section exécutée, les autres parties peuvent être lancées dans n’importe quel ordre.

Les sections suivantes peuvent être lancées dans n’importe quel ordre.

Sections d’exploration
Kinjal
Analyse des taux de commentaires et de réponses selon la note + premiers traitements NLP.

Julie
Répartition temporelle des commentaires (mois, saisons, années, tendances).

Laurine
Présence de réponse en fonction de la note et de la longueur des commentaires + analyse textuelle.

Quentin
Analyse de la longueur des commentaires vis-à-vis des notes et exploration lexicale.

## Preprocessing des données

02_preprocessing.ipynb

Ce notebook prépare les données pour toutes les étapes de modélisation (embeddings, topic modeling, classification).

Il transforme les données brutes en un jeu de données propre, structuré et réutilisable, et génère différents types d’embeddings.

### Étapes obligatoires (à exécuter en priorité)

Les deux sections suivantes doivent être lancées dans cet ordre :
1. Import et chargement des données
2. Preprocessing Général

### Preprocessing Général

Cette partie réalise les opérations suivantes :
- suppression des commentaires vides, trop courts ou dupliqués ;
- nettoyage textuel (minuscules, suppression des URLs, emails, mentions, espaces inutiles) ;
- détection et conservation uniquement des commentaires en français ;
- lemmatisation et suppression des stopwords linguistiques à l’aide de spaCy ;
- création de deux représentations distinctes du texte :
- texte avec contexte (pour les modèles BERT),
- texte sans contexte (pour les modèles statistiques et le topic modeling).

À l’issue de cette étape, un dataset final est sauvegardé et servira de base à toutes les modélisations.

### Génération des embeddings

Les sections suivantes du notebook concernent la génération des embeddings.
Elles peuvent être exécutées dans n’importe quel ordre, selon les besoins.

Embeddings disponibles :

- TF-IDF (avec sauvegarde du vectorizer),
- Sentence-BERT multilingue,
- CamemBERT (BERT français),
- Word2Vec (entraîné sur le corpus),
- (optionnel) modèles Word2Vec et FastText pré-entraînés.

Les embeddings sont sauvegardés dans le dossier data/embeddings afin d’éviter toute régénération inutile.

⚠️ Certaines cellules peuvent être longues à exécuter sans GPU, en particulier pour les modèles BERT.

### Top2Vec
Environnement Python recommandé
- Pour exécuter correctement le notebook, il est recommandé d'utiliser **Python 3.11.x**.  
- Les versions plus récentes (ex: Python 3.13) peuvent provoquer des erreurs d'import ou d'installation du package `top2vec`.  