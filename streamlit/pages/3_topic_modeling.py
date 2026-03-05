import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Projet Trustpilot",
    layout="wide"
)

st.title("Topic modeling")

st.text("Nous avons testé plusieurs modèles de topic modeling pour extraire les thèmes principaux des avis clients.")

st.subheader("Comparaison des modèles")

modeles = ["BERTopic", "Top2Vec", "CTM", "LDA"]
diversite = [0.78, 0.18, 0.80, 0.98]
coherence = [0.60, 0.39, 0.53, 0.35]

df = pd.DataFrame({
    "Modèle": modeles,
    "Diversité": diversite,
    "Cohérence c_v": coherence
})

st.dataframe(df)

st.divider()

st.subheader("Résultats par modèle")

@st.cache_data
def load_lda():
    df_LDA = pd.read_csv("./utils/LDA_topic_words.csv")
    LDA_topic_sizes = Image.open("./utils/LDA_topic_sizes.png")
    return df_LDA, LDA_topic_sizes

@st.cache_data
def load_top2vec():
    df_Top2Vec = pd.read_csv("./utils/Top2Vec_topic_words.csv")
    Top2Vec_topic_sizes = Image.open("./utils/Top2Vec_topic_sizes.png")
    Top2Vec_topic_proportion = Image.open("./utils/Top2Vec_topic_proportion.png")
    return df_Top2Vec, Top2Vec_topic_sizes, Top2Vec_topic_proportion

@st.cache_data
def load_ctm():
    df_CTM = pd.read_csv("./utils/CTM_topic_words.csv")
    CTM_topic_sizes = Image.open("./utils/CTM_topic_sizes.png")
    return df_CTM, CTM_topic_sizes

@st.cache_data
def load_bertopic():
    df_BERTopic = pd.read_csv("./utils/BERTopic_topic_words.csv")
    BERTopic_topic_sizes = Image.open("./utils/BERTopic_topic_sizes.png")
    BERTopic_word_scores = Image.open("./utils/BERTopic_word_scores.png")
    BERTopic_hierarchy = Image.open("./utils/BERTopic_hierarchy.png")
    return df_BERTopic, BERTopic_topic_sizes, BERTopic_word_scores, BERTopic_hierarchy

@st.cache_data
def load_resultat_final():
    topics_df = pd.read_excel("../data/labelled_topics/topics_top_words_phrases_annoté.xlsx")
    docs_df = pd.read_csv("../models/topic_modeling/bertopic/output/reviews_phrases_with_topics_final.csv")
    dataset_avis = pd.read_csv("../data/labelled_topics/dataset_avis.csv")
    dataset_phrases = pd.read_csv("../data/labelled_topics/dataset_phrases.csv")
    
    # Normalisation des catégories
    replace_map = {
        "Qualité Produit": "qualité produit",
        "Qualité produit": "qualité produit",
        "Service Livraison": "service livraison",
        "Service livraison": "service livraison",
        "Service Client": "service client",
        "Service client": "service client",
    }

    def normalize_category(cat):
        if pd.isna(cat):
            return None
        cat = cat.strip().lower()
        return replace_map.get(cat, cat)

    topics_df["Catégorie"] = topics_df["Catégorie"].apply(normalize_category)
    
    return topics_df, docs_df, dataset_avis, dataset_phrases

def run_resultat_final():
    st.subheader("Résultat final")
    st.text("Après plusieurs itérations et ajustements, nous avons obtenu un modèle BERTopic final qui extrait des topics plus cohérents et pertinents. Nous avons utilisé une approche de prétraitement des données plus rigoureuse, ainsi que des techniques de réduction de dimensionnalité pour améliorer la qualité des topics extraits.")

    topics_df, docs_df, dataset_avis, dataset_phrases = load_resultat_final()

    st.subheader("Affichage des données")

    with st.expander(f"Tableau des topics ({len(topics_df)} topics)"):
        st.dataframe(topics_df[["Topic", "Count", "Representation", "Representative_Docs", "Catégorie"]])
    
    with st.expander(f"Tableau des phrases avec topics ({len(docs_df)} phrases)"):
        st.dataframe(docs_df[["comment_id", "Commentaire", "sentence", "topics"]])
    
    with st.expander(f"Tableau des avis annotés ({len(dataset_avis)} avis)"):
        st.dataframe(dataset_avis[["comment_id", "Commentaire", "clean_comment", "qualité produit", "service livraison", "service client"]])
    
    with st.expander(f"Tableau des phrases annotées ({len(dataset_phrases)} phrases)"):
        st.dataframe(dataset_phrases[["comment_id", "Commentaire", "clean_comment", "sentence", "topics", "Catégorie", "qualité produit", "service livraison", "service client"]])
    
    st.divider()
    
    # ---------- REPARTITION CLASSES ----------
    st.subheader("Répartition des classes")

    topic_to_cat = dict(zip(topics_df["Topic"], topics_df["Catégorie"]))

    valid_labels = ["qualité produit", "service livraison", "service client"]

    docs_df_copy = docs_df.copy()
    docs_df_copy["Catégorie"] = docs_df_copy["topics"].map(topic_to_cat)

    docs_df_copy["Label final"] = docs_df_copy["Catégorie"].apply(
        lambda x: x if x in valid_labels else "aucun label"
    )

    labels_order = [
        "qualité produit",
        "service livraison",
        "service client",
        "aucun label"
    ]

    # ---------- PHRASES ----------
    phrase_counts = (
        docs_df_copy["Label final"]
        .value_counts()
        .reindex(labels_order, fill_value=0)
    )

    # ---------- COMMENTAIRES ----------
    comment_labels = (
        docs_df_copy
        .groupby("comment_id")["Label final"]
        .apply(lambda x: set(x))
    )

    counts = {
        "qualité produit": 0,
        "service livraison": 0,
        "service client": 0,
        "aucun label": 0
    }

    for labels in comment_labels:
        valid = [l for l in labels if l != "aucun label"]

        if len(valid) == 0:
            counts["aucun label"] += 1
        else:
            for l in valid:
                counts[l] += 1

    comment_counts = pd.Series(counts).reindex(labels_order)

    # ---------- ECHELLE COMMUNE ----------
    max_y = max(phrase_counts.max(), comment_counts.max()) * 1.1

    col1, col2 = st.columns(2)

    # ---------- GRAPHIQUE PHRASES ----------
    with col1:
        st.markdown("##### Répartition des classes (phrases)")

        fig, ax = plt.subplots()
        phrase_counts.plot(kind="bar", ax=ax)

        ax.set_ylim(0, max_y)
        ax.set_xlabel("Label")
        ax.set_ylabel("Nombre de phrases")
        ax.set_title("Distribution des labels - phrases")

        st.pyplot(fig)

    # ---------- GRAPHIQUE COMMENTAIRES ----------
    with col2:
        st.markdown("##### Répartition des classes (commentaires)")

        fig2, ax2 = plt.subplots()
        comment_counts.plot(kind="bar", ax=ax2)

        ax2.set_ylim(0, max_y)
        ax2.set_xlabel("Label")
        ax2.set_ylabel("Nombre de commentaires")
        ax2.set_title("Distribution des labels - commentaires")

        st.pyplot(fig2)

        st.info("Un commentaire peut être associé à plusieurs labels à la fois.")
        
        
    st.divider()
    
    # ---------- EXPLORATION TOPIC ----------
    st.subheader("Explorer un topic")

    topic_id = st.selectbox(
        "Choisir un topic",
        topics_df["Topic"]
    )

    topic_words = topics_df.loc[
        topics_df.Topic == topic_id,
        "Representation"
    ].values[0]
    
    st.markdown(f"**Mots clés :** {topic_words[1: -1]}")
    
    categorie = topics_df.loc[topics_df.Topic == topic_id, "Catégorie"].values[0]
    st.markdown(f"**Catégorie :** {categorie}")
    
    if categorie in ["qualité produit", "service livraison", "service client"]:
        st.markdown(f"**Label final :** {categorie}")
    else:
        st.markdown("**Label final :** Aucun label")

    st.markdown("##### Phrases associées")

    subset = docs_df[docs_df.topics == topic_id]

    st.dataframe(
        subset[["Commentaire", "sentence", "comment_id", "star"]].head(50),
        width='stretch'
        )

    st.divider()

    # ---------- EXPLORATION COMMENTAIRE ----------
    st.subheader("Explorer un commentaire")

    comment_id = st.selectbox(
        "Choisir un commentaire",
        docs_df["comment_id"].unique()
    )
    
    comment = docs_df[docs_df["comment_id"] == comment_id]["Commentaire"].values[0]
    st.markdown(f"**Commentaire :** {comment}")
    
    topic_to_cat = dict(zip(topics_df["Topic"], topics_df["Catégorie"]))
    
    comment_subset = docs_df[docs_df.comment_id == comment_id]
    comment_subset = comment_subset.copy()
    comment_subset["Catégorie"] = comment_subset["topics"].map(topic_to_cat)
    
    valid_labels = ["qualité produit", "service livraison", "service client"]

    comment_subset["Label final"] = comment_subset["Catégorie"].apply(
        lambda x: x if x in valid_labels else None
    )
    
    cats = (
    comment_subset["topics"]
    .map(topic_to_cat)
    .dropna()
    .unique()
    )
    
    st.markdown("##### Catégories détectées")
    
    st.markdown("\n".join(f"- {item}" for item in cats))
    
    labels_finaux = (
    comment_subset["Label final"]
    .dropna()
    .unique()
    )
    
    st.markdown("##### Labels finaux")

    if len(labels_finaux) > 0:
        st.markdown("\n".join(f"- {item}" for item in labels_finaux))
    else:
        st.markdown("Aucun label")
        
    st.markdown("##### Phrases du commentaire")

    st.dataframe(
        comment_subset[["sentence", "Catégorie", "Label final", "star"]],
        width='stretch'
    )
    
    
    # AFFICHER LA REPARTITION DES LABELS FINAUX DANS LE DATASET
    # AFFICHER LE LABEL FINAL DANS L'EXPLORATION D'UN COMMENTAIRE EN PLUS DE LA CATEGORIE






cols = st.columns(3)
with cols[0]:
    model = st.selectbox("Sélectionnez un modèle de topic modeling", ("LDA", "Top2Vec", "CTM", "BERTopic", "Résultat final"))

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if st.button("Valider"):
    st.session_state.selected_model = model
    
if st.session_state.selected_model == "LDA":
    st.subheader("LDA (Latent Dirichlet Allocation)")
    st.text("LDA est un modèle de topic modeling probabiliste qui suppose que les documents sont des mélanges de topics et que les topics sont des mélanges de mots. Il utilise une approche bayésienne pour inférer les distributions de topics et de mots.")
    
    df_LDA, LDA_topic_sizes = load_lda()
    
    st.dataframe(df_LDA)
    st.image(LDA_topic_sizes)
elif st.session_state.selected_model == "Top2Vec":
    st.subheader("Top2Vec")
    st.text("Top2Vec est un modèle de topic modeling qui utilise des embeddings de mots pour regrouper les documents similaires en topics. Il est capable de trouver des topics de manière non supervisée et peut gérer de grandes quantités de données.")
    
    df_Top2Vec, Top2Vec_topic_sizes, Top2Vec_topic_proportion = load_top2vec()
    
    st.dataframe(df_Top2Vec)
    st.image(Top2Vec_topic_sizes)
    st.image(Top2Vec_topic_proportion)
elif st.session_state.selected_model == "CTM":
    st.subheader("CTM (Combined Topic Model)")
    st.text("CTM est un modèle de topic modeling qui combine les avantages de LDA et de Top2Vec. Il utilise des embeddings de mots pour regrouper les documents similaires en topics, tout en conservant une structure probabiliste pour inférer les distributions de topics et de mots.")
    
    df_CTM, CTM_topic_sizes = load_ctm()
    
    st.dataframe(df_CTM)
    st.image(CTM_topic_sizes)
elif st.session_state.selected_model == "BERTopic":
    st.subheader("BERTopic")
    st.text("BERTopic est un modèle de topic modeling basé sur des embeddings de phrases. Il utilise des techniques de réduction de dimensionnalité pour regrouper les avis similaires en topics.")
    
    df_BERTopic, BERTopic_topic_sizes, BERTopic_word_scores, BERTopic_hierarchy = load_bertopic()
    
    st.dataframe(df_BERTopic)
    st.image(BERTopic_topic_sizes)
    st.image(BERTopic_word_scores)
    st.image(BERTopic_hierarchy)

elif st.session_state.selected_model == "Résultat final":
    run_resultat_final()


