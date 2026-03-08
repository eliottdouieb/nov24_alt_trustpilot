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
st.info(f"Dataset avant nettoyage : {len(df)} commentaires.")
st.write("Dans cette partie, nous préparons les commentaires avant d'entraîner les modèles.")
st.header("1. Nettoyage des données")
with st.expander("Voir les étapes de nettoyage"):
    st.write( """
    - suppression des commentaires vides, trop courts ou dupliqués     
    - Suppression des URLs et emails  
    - Mise en minuscules  
    - Détection de la langue française  
    - Lemmatisation avec SpaCy  
    - Suppression des stopwords  
    - Suppression des mots liés au produit (stopwords custom)
            """ )

def count_words(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    return len(text.split())

df = df.dropna(subset=["Commentaire"])
df = df.drop_duplicates(subset=["Commentaire", "client"], keep="first")
df = df[df["Commentaire"].apply(count_words) > 3].reset_index(drop=True)

# Fonction de nettoyage 
def clean_text(text):
    # Gérer les Nan/float
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"(http|www)\S+", "", text) # URLs
    text = re.sub(r"\S+@\S+", "", text) # emails
    text = re.sub(r"(@|#)\w+", "", text) # mentions @ et #
    text = re.sub(r"\s+", " ", text).strip() # espaces multiples, sauts de lignes , tabulations
    return text
df["clean_comment"] = df["Commentaire"].apply(clean_text)

# Suppression des commentaires pas en français
DetectorFactory.seed = 42

def is_french(text):
    try:
        return detect(text) == "fr"
    except LangDetectException:
        return False

df = df[df["clean_comment"].apply(is_french)].reset_index(drop=True)

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

st.subheader("Choisir un commentaire du dataset")

index = st.number_input(
    "Veuillez saisir un numéro d'index pour afficher dynamiquement un commentaire du dataset.",
    min_value=0,
    max_value=len(df)-1,
    value=7
)
# Récupérer le commentaire
text = df["Commentaire"].iloc[index]

# Afficher le commentaire sélectionné
#st.subheader("Sélectionner le commentaire original dans le dataset.")
#st.write(text) 

# Ensuite appliquer le preprocessing
original = df["Commentaire"].iloc[index]
step1 = clean_text(text)
tokens = preprocess_spacy(step1)
step2 = " ".join(tokens)
step3 = " ".join([t for t in tokens if t not in STOPWORDS_CUSTOM]) 
st.markdown("Transformation progressive du commentaire :")

with st.expander("1️⃣ Texte original"):
    st.write(original)

with st.expander("2️⃣ Après nettoyage regex"):
    st.write(step1)

with st.expander("3️⃣ Après lemmatisation + suppression stopwords"):
    st.write(step2)

with st.expander("4️⃣ Après suppression stopwords personnalisés"):
    st.write(step3)

st.info(f"Dataset après nettoyage : {len(df)} commentaires.")
st.header("3. Transformation du texte en vecteurs (Embeddings)")
st.write("""
Après le nettoyage, les commentaires sont transformés en vecteurs numériques 
afin d’être utilisés par les modèles de machine learning.
""")
# Création du tableau
embeddings_data = {
    "Méthode": ["TF-IDF", "Word2Vec", "Sentence-BERT", "CamemBERT"],
    "Description": [
        "Méthode statistique qui donne plus d’importance aux mots spécifiques et moins d’importance aux mots fréquents dans le corpus.",
        "Modèle qui apprend les relations entre les mots.",
        "Modèle avancé qui comprend le sens global des phrases.",
        "Modèle BERT spécialisé pour le français."
    ]
}
df_embeddings = pd.DataFrame(embeddings_data)

st.table(df_embeddings)


