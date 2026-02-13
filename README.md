# Trust Pilot

Projet de topic modeling et de classification de commentaires Trustpilot à partir de techniques NLP classiques et contextuelles
==============================

## Objectif du projet
Ce projet a pour objectif d'ajouter une nouvelle fonctionnalité au site Trustpilot : la possibilité de trier les commentaires d'une entreprise par thème.

Pour cela, le projet s'effectue en plusieurs étapes :
- Regrouper les commentaires par thèmes grâce au topic modeling.
- Regrouper et nommer les topics obtenus à l'aide d'une annotation manuelle.
- A partir des topics cohérents obtenus, labelliser les commentaires restants en utilisant un modèle de classification supervisée multilabel.
- (Optionnel) A l'aide des données labellisées obtenues, créer un modèle de classification supervisée capable de prédire la catégorie d'un nouvel avis.

## Installation et mise en place

1️. Cloner le dépôt

```bash
git clone https://github.com/eliottdouieb/nov24_alt_trustpilot
cd nov24_alt_trustpilot
```
2️. Créer un environnement virtuel (fortement recommandé)

Sur Windows :

```bash
python -m venv venv
venv\Scripts\activate
```

Sur Linux :

```bash
python -m venv venv
source venv/bin/activate
```

3️. Installer les dépendances

```bash
pip install -r requirements.txt
```

Prérequis
- Python ≥ 3.10 (3.11.9 recommandé)
- (fortement recommandé) un environnement virtuel Python


## Organisation du projet

Le projet est structuré autour de plusieurs notebooks Jupyter :

- 01_exploration_de_donnees.ipynb
- 02_preprocessing.ipynb
- 03_topicmodeling_*.ipynb (plusieurs approches de Topic Modeling testées)
- 04_classification_*.ipynb (plusieurs modèles de Classification testés)

👉 Le notebook d’exploration est optionnel.

👉 Le notebook de preprocessing est obligatoire avant toute modélisation.

## Exécution d'un notebook

Création du kernel pour Jupyter :

```bash
python -m ipykernel install --user --name=venv --display-name "TrustPilot Venv"
```

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

### Sections d’exploration

- Kinjal : Analyse des taux de commentaires et de réponses selon la note + premiers traitements NLP.
- Julie : Répartition temporelle des commentaires (mois, saisons, années, tendances).
- Laurine : Présence de réponse en fonction de la note et de la longueur des commentaires + analyse textuelle.
- Quentin : Analyse de la longueur des commentaires vis-à-vis des notes et exploration lexicale.

## Preprocessing des données

02_preprocessing.ipynb

Ce notebook prépare les données pour toutes les étapes de modélisation (embeddings, topic modeling, classification).

Il transforme les données brutes en un jeu de données propre, structuré et réutilisable, et génère différents types d’embeddings.

### Étapes obligatoires (à exécuter en priorité)

Les deux sections suivantes doivent être lancées dans cet ordre :
1. Import et chargement des données
2. Preprocessing Général

### Preprocessing Général

téléchargez le modèle linguistique français de spaCy :
```bash
python -m spacy download fr_core_news_md
```
Ce modèle est utilisé pour la lemmatisation et la suppression des stopwords.

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

#### Téléchargement des modèles Word2Vec et FastText pré-entraînés

Pour utiliser ces modèles, il faut d'abord les télécharger :
- [Word2Vec](https://fauconnier.github.io/#data)
- [Fasttext français binaire](https://fasttext.cc/docs/en/crawl-vectors.html)

## Topic Modeling et Classification

Les étapes de Topic Modeling (`03_`) et de Classification (`04_`) sont liées par une étape d'annotation manuelle.

**Important :**
L'output des notebooks `03_topicmodeling` sert de base à l'annotation, qui produit ensuite l'input pour les notebooks `04_classification` (le code est disponible dans le notebook `04_classification_XGBoost`).

Cependant, **les données annotées nécessaires pour l'étape 4 sont déjà incluses dans le dépôt git.**

Conséquences :
- Vous **pouvez** exécuter les notebooks de classification (`04`) directement, **sans avoir exécuté** les notebooks de topic modeling (`03`).
- Si vous relancez l'étape 3, vous ne pourrez pas utiliser directement vos résultats pour l'étape 4 sans refaire l'annotation manuelle.

### Top2Vec

Environnement Python recommandé
- Pour exécuter correctement le notebook, il est recommandé d'utiliser **Python 3.11.9**.  
- Les versions plus récentes (ex: Python 3.13) peuvent provoquer des erreurs d'import ou d'installation du package `top2vec`.

### Alternative : Annotation automatique (Weakly Supervised)

Nous avons essayé une alternative au topic modeling qui est complètement automatique : une **annotation weakly-supervisée par regex**.

Pour mettre en place cette alternative :
1. Exécutez le notebook `03_Labellisation_Regex.ipynb`.
2. La classification LogisticRegression est flexible. Dans la première cellule du notebook `04_classification_LogisticRegression.ipynb`, modifiez la variable `ENTRAINEMENT` :
   ```python
   ENTRAINEMENT = 'regex'
   ```
   Cela utilisera automatiquement le dataset annoté par regex pour cette classification.

Ainsi, pour le classeur Logistic Regression, vous pouvez choisir l'un ou l'autre comme données d'entraînement :
- **Topic Modeling** (annotation manuelle)
- **Regex** (annotation automatique)

## Equipe du projet

- Julie BOUTELET
- Laurine CHARBONNIER
- Quentin GEORGE
- Kinjal KAPADIA