import streamlit as st
import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException
from pathlib import Path
import spacy
import re

st.set_page_config(
    page_title="Projet Trustpilot",
    layout="wide"
)
PROJECT_ROOT = Path.cwd().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

df = pd.read_csv(DATA_DIR / "reviews_trust.csv")

st.title("Preprocessing")
st.write("Dans cette partie, nous préparons les commentaires avant d'entraîner les modèles.")
st.header("1. Nettoyage des données")
with st.expander("Voir les étapes de nettoyage"):
    st.write( """
        - suppression des commentaires vides, trop courts ou dupliqués ;
        - nettoyage textuel (minuscules, suppression des URLs, emails, mentions, espaces inutiles) ;
        - détection et conservation uniquement des commentaires en français ;
        - lemmatisation et suppression des stopwords linguistiques à l’aide de spaCy ;
        - création de deux représentations distinctes du texte :
        - texte avec contexte (pour les modèles BERT),
        - texte sans contexte (pour les modèles statistiques et le topic modeling).
            """ )
# Le modèle spaCy est chargé une seule fois grâce à `@st.cache_resource` afin d’optimiser les performances de l’application.
# La fonctionnalité ner (reconnaissance des entités nommées) est désactivée car elle n’est pas nécessaire pour la classification.
@st.cache_resource
def load_spacy():
    return spacy.load("fr_core_news_md", disable=["ner"])
nlp = load_spacy()

#Des stopwords personnalisés (liés aux produits) sont également retirés pour éviter un biais dans l’analyse.
STOPWORDS_CUSTOM = {
    'montre', 'montres', 'boucle', 'boucles', 'oreil', 'les', 'oreille', 'paire', 'paires', 'lunettes', 'bracelet', 'bracelets', 'collier', 'iphone', 'téléphone', 'pendentif', 'robot', 'robots', 'aspirateur', 'baskets', 'basket', 'chaussure', 'chaussures', 'sandales', 'plantes', 'plante', 'arbres', 'arbre', 'bulbes', 'willemse', 'sommiers', 'jardin', 'bague', 'lampe', 'lampes', 'abat-jour', 'abat jour', 'lampadaire', 'parfum', 'shampoing', 'shampooing', 'shampooings', 'cheveux', 'masque', 'masques', 'crème', 'élastiques', 'manteau', 'bougie', 'cadre', 'écouteurs', 'vélo', 'robe', 'vêtements', 'bijoux', 'sac', 'portable', 'clio', 'luminaire', 'oreillette', 'induction', 'écouteur', 'couette', 'samsung', 'téléphones', 'smartcase', 'abat', 'apple', 'watch', 'shirt', 'tee', 'chemise', 'shirts', 'hortensias', 'orchidée', 'sacs', 'plant', 'chaises', 'lacoste', 'polo', 'pantalon', 'jean', 'jeans', 'sneakers', 'lunette', 'écran', 'tablette', 'tablettes', 'table', 'tables', 'chemisier', 'pulls', 'pull', 'trotinettes', 'trotinette', 'chaussons', 'chausson', 'brosse', 'brosses', 'crèmes', 'gel', 'gels', 'parfums', 'robe', 'robes', 'sacoches', 'sacoche', 'vestes', 'veste'
}

st.header("2. Avant / Après nettoyage")
st.write("""
Cette section montre l’impact concret des différentes étapes 
de preprocessing sur un commentaire.
""")
# Fonction de nettoyage 
def clean_text(text):
    # Gérer les Nan/float
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"(http|www)\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"(@|#)\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Nous utilisons la lemmatisation pour ramener les mots à leur forme de base.
# Les stopwords standards sont supprimés automatiquement par spaCy.

def preprocess_spacy(text):
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop and len(token) > 2
    ]
    return tokens

default_text = df["Commentaire"].iloc[7]
text = st.text_area(
    "Exemple de commentaire",
    value=default_text
    )

step1 = clean_text(text)
tokens = preprocess_spacy(step1)
step2 = " ".join(tokens)
step3 = " ".join(
    t for t in tokens if t not in STOPWORDS_CUSTOM
)
st.subheader("Commentaire original")
st.write(text)

st.subheader("Après nettoyage regex")
st.write(step1)

st.subheader("Après lemmattisation + suppression stopwords")
st.write(step2)

st.subheader("Après suppression stopwords custom")
st.write(step3) 

#Création de la colonne nettoyée 
#df["clean_comment"] = df["Commentaire"].apply(clean_text)

#st.subheader("Exemple de nettoyage sur le dataset")

#st.dataframe(
#    df[["Commentaire", "clean_comment"]].head(5)
#)

st.header("3. Transformation du texte en vecteurs (Embeddings)")
st.write("""
Après le nettoyage, les commentaires sont transformés en vecteurs numériques 
afin d’être utilisés par les modèles de machine learning.
""")
# Création du tableau
embeddings_data = {
    "Méthode": ["TF-IDF", "Word2Vec", "Sentence-BERT", "CamemBERT"],
    "Description": [
        "Méthode statistique qui identifie les mots les plus importants dans un texte.",
        "Modèle qui apprend les relations entre les mots.",
        "Modèle avancé qui comprend le sens global des phrases.",
        "Modèle BERT spécialisé pour le français."
    ]
}
df_embeddings = pd.DataFrame(embeddings_data)

st.table(df_embeddings)


