import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(
    page_title="Projet Trustpilot",
    layout="wide"
)

st.title("Topic modeling")

st.text("Nous avons testé plusieurs modèles de topic modeling pour extraire les thèmes principaux des avis clients.")

st.header("Comparaison des modèles")

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

st.header("Résultats par modèle")

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

@st.cache_data
def load_lda():
    df_LDA = pd.read_csv(UTILS_DIR / "LDA_topic_words.csv")
    LDA_topic_sizes = Image.open(UTILS_DIR / "LDA_topic_sizes.png")
    return df_LDA, LDA_topic_sizes

@st.cache_data
def load_top2vec():
    df_Top2Vec = pd.read_csv(UTILS_DIR / "Top2Vec_topic_words.csv")
    Top2Vec_topic_sizes = Image.open(UTILS_DIR / "Top2Vec_topic_sizes.png")
    Top2Vec_topic_proportion = Image.open(UTILS_DIR / "Top2Vec_topic_proportion.png")
    return df_Top2Vec, Top2Vec_topic_sizes, Top2Vec_topic_proportion

@st.cache_data
def load_ctm():
    df_CTM = pd.read_csv(UTILS_DIR / "CTM_topic_words.csv")
    CTM_topic_sizes = Image.open(UTILS_DIR / "CTM_topic_sizes.png")
    return df_CTM, CTM_topic_sizes

@st.cache_data
def load_bertopic():
    df_BERTopic = pd.read_csv(UTILS_DIR / "BERTopic_topic_words.csv")
    BERTopic_topic_sizes = Image.open(UTILS_DIR / "BERTopic_topic_sizes.png")
    BERTopic_word_scores = Image.open(UTILS_DIR / "BERTopic_word_scores.png")
    BERTopic_hierarchy = Image.open(UTILS_DIR / "BERTopic_hierarchy.png")
    return df_BERTopic, BERTopic_topic_sizes, BERTopic_word_scores, BERTopic_hierarchy

@st.cache_data
def load_resultat_final():
    topics_df = pd.read_excel(DATA_DIR / "labelled_topics/topics_top_words_phrases_annoté.xlsx")
    docs_df = pd.read_csv(MODEL_DIR / "topic_modeling/bertopic/output/reviews_phrases_with_topics_final.csv")
    dataset_avis = pd.read_csv(DATA_DIR / "labelled_topics/dataset_avis.csv")
    dataset_phrases = pd.read_csv(DATA_DIR / "labelled_topics/dataset_phrases.csv")
    
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
    st.header("Résultat final")
    st.text("Après plusieurs itérations et ajustements, nous avons obtenu un modèle BERTopic final qui extrait des topics plus cohérents et pertinents. Nous avons utilisé une approche de prétraitement des données plus rigoureuse, ainsi que des techniques de réduction de dimensionnalité pour améliorer la qualité des topics extraits.")

    topics_df, docs_df, dataset_avis, dataset_phrases = load_resultat_final()

    st.header("Affichage des données")

    with st.expander(f"Tableau des topics"):
        st.dataframe(topics_df[["Topic", "Count", "Representation", "Representative_Docs", "Catégorie"]])
    
    with st.expander(f"Tableau des phrases avec topics"):
        st.dataframe(docs_df[["comment_id", "Commentaire", "sentence", "topics"]])
    
    with st.expander(f"Tableau des avis annotés"):
        st.dataframe(dataset_avis[["comment_id", "Commentaire", "clean_comment", "qualité produit", "service livraison", "service client"]])
    
    with st.expander(f"Tableau des phrases annotées"):
        st.dataframe(dataset_phrases[["comment_id", "Commentaire", "clean_comment", "sentence", "topics", "Catégorie", "qualité produit", "service livraison", "service client"]])
    
    st.divider()
    
    # ---------- STATISTIQUES ----------
    
    st.header("Statistiques")
    
    valid_cats = ["qualité produit", "service livraison", "service client"]

    topic_to_cat_map = dict(zip(topics_df["Topic"], topics_df["Catégorie"]))

    df_temp = docs_df.copy()
    df_temp["categorie_detectee"] = df_temp["topics"].map(topic_to_cat_map)

    df_temp["label_final"] = df_temp["categorie_detectee"].apply(
        lambda x: x if x in valid_cats else None
    )
    
    labels_par_commentaire = (
        df_temp
        .groupby("comment_id")["label_final"]
        .apply(lambda x: list(set(x.dropna())))
    )
    
    st.subheader("Statistiques par commentaires")
    
    nb_commentaires = len(labels_par_commentaire)

    commentaires_classes = labels_par_commentaire.apply(len).gt(0).sum()

    taux_couverture = (commentaires_classes / nb_commentaires) * 100
    
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Total des commentaires", f"{nb_commentaires:,}".replace(",", " "))
    m2.metric("Commentaires classés", f"{commentaires_classes:,}".replace(",", " "))
    m3.metric("Commentaires non classés", f"{nb_commentaires - commentaires_classes:,}".replace(",", " "))
    m4.metric("Taux de couverture", f"{taux_couverture:.1f}%")
    
    st.subheader("Statistiques par phrases")
    
    total_phrases = len(df_temp)

    phrases_classees = df_temp["label_final"].notna().sum()

    couverture_phrases = (phrases_classees / total_phrases) * 100

    p1, p2, p3, p4 = st.columns(4)

    p1.metric("Total des phrases", f"{total_phrases:,}".replace(",", " "))
    p2.metric("Phrases classées", f"{phrases_classees:,}".replace(",", " "))
    p3.metric("Phrases non classées", f"{total_phrases - phrases_classees:,}".replace(",", " "))
    p4.metric("Taux de couverture", f"{couverture_phrases:.1f}%")
    
    st.subheader("Répartition des catégories détectées")

    flat_labels = labels_par_commentaire.explode()

    distribution_labels = (
        flat_labels
        .value_counts()
        .reindex(valid_cats, fill_value=0)
        .reset_index()
    )

    distribution_labels.columns = ["Catégorie", "Nombre de commentaires"]
    
    phrase_labels = (
        df_temp["label_final"]
        .dropna()
        .value_counts()
        .reindex(valid_cats, fill_value=0)
        .reset_index()
    )

    phrase_labels.columns = ["Catégorie", "Nombre de phrases"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(
            distribution_labels,
            x="Nombre de commentaires",
            y="Catégorie",
            orientation="h",
            text_auto=True,
            title="Nombre de commentaires par catégorie",
            color="Catégorie",
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
        )

        fig_bar.update_layout(
            showlegend=False,
            yaxis_title=None
        )

        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col2:
        fig_bar_phrases = px.bar(
            phrase_labels,
            x="Nombre de phrases",
            y="Catégorie",
            orientation="h",
            text_auto=True,
            title="Nombre de phrases par catégorie",
            color="Catégorie",
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
        )

        fig_bar_phrases.update_layout(
            showlegend=False,
            yaxis_title=None
        )

        st.plotly_chart(fig_bar_phrases, use_container_width=True)
    
    st.divider()
    
    # ---------- EXPLORATION TOPIC ----------
    st.header("Explorer un topic")

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

    st.subheader("Phrases associées")

    subset = docs_df[docs_df.topics == topic_id]

    st.dataframe(
        subset[["Commentaire", "sentence", "comment_id", "star"]].head(50),
        width='stretch'
        )

    st.divider()

    # ---------- EXPLORATION COMMENTAIRE ----------
    st.header("Explorer un commentaire")

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
    
    st.subheader("Catégories détectées")
    
    st.markdown("\n".join(f"- {item}" for item in cats))
    
    labels_finaux = (
    comment_subset["Label final"]
    .dropna()
    .unique()
    )
    
    st.subheader("Labels finaux")

    if len(labels_finaux) > 0:
        st.markdown("\n".join(f"- {item}" for item in labels_finaux))
    else:
        st.markdown("Aucun label")
        
    st.subheader("Phrases du commentaire")

    st.dataframe(
        comment_subset[["sentence", "Catégorie", "Label final", "star"]],
        width='stretch'
    )





cols = st.columns(3)
with cols[0]:
    st.session_state.selected_model = st.selectbox(
        "Sélectionnez un modèle de topic modeling", 
        ("LDA", "Top2Vec", "CTM", "BERTopic", "Résultat final"), 
        index=4
    )

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if st.session_state.selected_model == "LDA":
    st.header("LDA (Latent Dirichlet Allocation)")
    st.text("LDA est un modèle de topic modeling probabiliste qui suppose que les documents sont des mélanges de topics et que les topics sont des mélanges de mots. Il utilise une approche bayésienne pour inférer les distributions de topics et de mots.")
    
    df_LDA, LDA_topic_sizes = load_lda()
    
    st.dataframe(df_LDA)
    st.image(LDA_topic_sizes)
elif st.session_state.selected_model == "Top2Vec":
    st.header("Top2Vec")
    st.text("Top2Vec est un modèle de topic modeling qui utilise des embeddings de mots pour regrouper les documents similaires en topics. Il est capable de trouver des topics de manière non supervisée et peut gérer de grandes quantités de données.")
    
    df_Top2Vec, Top2Vec_topic_sizes, Top2Vec_topic_proportion = load_top2vec()
    
    st.dataframe(df_Top2Vec)
    st.image(Top2Vec_topic_sizes)
    # st.image(Top2Vec_topic_proportion)
elif st.session_state.selected_model == "CTM":
    st.header("CTM (Combined Topic Model)")
    st.text("CTM est un modèle de topic modeling qui combine les avantages de LDA et de Top2Vec. Il utilise des embeddings de mots pour regrouper les documents similaires en topics, tout en conservant une structure probabiliste pour inférer les distributions de topics et de mots.")
    
    df_CTM, CTM_topic_sizes = load_ctm()
    
    st.dataframe(df_CTM)
    st.image(CTM_topic_sizes)
elif st.session_state.selected_model == "BERTopic":
    st.header("BERTopic")
    st.text("BERTopic est un modèle de topic modeling basé sur des embeddings de phrases. Il utilise des techniques de réduction de dimensionnalité pour regrouper les avis similaires en topics.")
    
    df_BERTopic, BERTopic_topic_sizes, BERTopic_word_scores, BERTopic_hierarchy = load_bertopic()
    
    st.dataframe(df_BERTopic)
    st.image(BERTopic_topic_sizes)
    # st.image(BERTopic_word_scores)
    # st.image(BERTopic_hierarchy)

elif st.session_state.selected_model == "Résultat final":
    run_resultat_final()


