import streamlit as st
import pandas as pd
from pathlib import Path
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

plt.rcParams["figure.figsize"] = (4,3)   # taille plus petite pour tous les graphiques
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8

st.set_page_config(
    page_title="Projet Trustpilot",
    layout="wide"
)

st.header("Exploration des données")

# Load your dataset
# Example: a CSV file in your project folder
BASE_DIR = Path(__file__).resolve().parents[2]  # remonte à la racine du projet
DATA_PATH = BASE_DIR / "data" / "raw" / "reviews_trust.csv"

df = pd.read_csv(DATA_PATH)

# AFFICHER UN HEAD AVEC LES PREMIERES LIGNES DU DATASET
st.title(f"1. Affichage du dataset")
st.dataframe(df.head(), width="stretch")

counts = df["company"].value_counts().reset_index()
counts.columns = ["company", "count"]

st.dataframe(counts)

# DIAGRAMME NBR COMMENTAIRES PAR NOTE
# Comptage des avis par note
counts = df["star"].value_counts().sort_index()
total_comments = counts.sum()

st.title(f"2. Répartition du nombre de commentaires par note (Total = {total_comments})")

# Création du graphique
fig, ax = plt.subplots()
ax.bar(counts.index, counts.values, color="skyblue")

ax.set_xlabel("Note")
ax.set_ylabel("Nombre de commentaires")
ax.set_title("")  # titre vide car déjà dans st.title

st.pyplot(fig, use_container_width=False)

# ANALYSE DES WORDCLOUDS PAR NOTE
st.title("3. Analyse des WordClouds par note")

#FRENCH STOPWORDS
def ensure_nltk_resource(resource):
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

ensure_nltk_resource("stopwords")

FRENCH_STOPWORDS = stopwords.words("french")

# clean de la colonne "commentaire"
stop_fr = set(FRENCH_STOPWORDS)
import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ ]", " ", text)  # garder lettres et accents
    words = [w for w in text.split() if w not in stop_fr and len(w) > 2]
    return " ".join(words)

df["comment_clean"] = df["Commentaire"].fillna("").apply(clean_text)

# Vérification que les colonnes existent
if "star" in df.columns and "comment_clean" in df.columns:

    # Sélection de la note
    note_choisie = st.selectbox(
        "Choisissez une note :",
        sorted(df["star"].unique())
    )

    # Filtrage
    df_filtre = df[df["star"] == note_choisie]

    st.write(f"Nombre d'avis pour la note {note_choisie} ⭐ :", len(df_filtre))

    comments = " ".join(df_filtre["comment_clean"].dropna())

    # Génération du WordCloud
    if comments.strip():
        wc = WordCloud(
            width=600,
            height=400,
            background_color="white"
        ).generate(comments)

        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig, use_container_width=False)
    else:
        st.warning("Aucun commentaire disponible pour cette note.")

else:
    st.error("Les colonnes 'star' ou 'comment_clean' sont manquantes dans le dataframe.")

# AFFICHAGE DES TOPS N-GRAM    
# Nettoyage simple des textes
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
    else:
        text = ""
    return text

df["clean_comment"] = df["Commentaire"].apply(clean_text)


st.title("4. Analyse des n-grams et répartition des notes")
stop_words = set(FRENCH_STOPWORDS)

# --- Étape 1 : Calcul des top n-grams ---
# df["clean_comment"] et df["star"] doivent exister
vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words=list(stop_words))  # bigrammes + trigrammes
X = vectorizer.fit_transform(df["clean_comment"])

# Comptage global
sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# DataFrame top 20
top_ngrams = pd.DataFrame(words_freq[:20], columns=["ngram", "count"])
st.subheader("Top 20 des bigrammes les plus fréquents")
st.dataframe(top_ngrams)

# --- Étape 2 : Sélection d'un n-gram pour afficher la répartition des notes ---
top_ngrams_list = top_ngrams["ngram"].tolist()
ngram_selectionne = st.selectbox("Choisissez un n-gram :", top_ngrams_list)

# Filtrer les commentaires contenant le n-gram choisi
mask = df["clean_comment"].str.contains(ngram_selectionne, case=False, na=False)
subset = df[mask]

if subset.empty:
    st.warning("Aucun commentaire pour ce n-gram.")
else:
    star_counts = subset["star"].value_counts().sort_index()

    # --- Étape 3 : Graphique de la répartition des notes ---
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(
        x=star_counts.index,
        y=star_counts.values,
        ax=ax,
        color="#4C72B0"
    )
    ax.set_xlabel("Note")
    ax.set_ylabel("Nombre d'avis")
    ax.set_title(f"Répartition des notes pour : '{ngram_selectionne}'")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    st.pyplot(fig, use_container_width=False)


# DIAGRAMME LONGUEUR MOYENNE DES COMMENTAIRES

st.title("5. Longueur moyenne des commentaires par note")

# Nettoyage des données
df_clean = df.dropna(subset=["Commentaire"]).copy()
df_clean["longueur"] = df_clean["Commentaire"].str.len()

# Création de la figure
fig, ax = plt.subplots()

sns.violinplot(
    x="star",
    y="longueur",
    data=df_clean,
    inner="box",
    density_norm="width",
    ax=ax
)

ax.set_xlabel("Note")
ax.set_ylabel("Longueur du commentaire")

# Affichage dans Streamlit
st.pyplot(fig, use_container_width=False)
