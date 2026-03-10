
import streamlit as st

st.set_page_config(
    page_title="Projet Trustpilot",
    layout="wide"
)

# Logo Trustpilot
st.image("images/TrustPilot.png", width=500)

# Objectif du projet
st.header("Objectif du projet")

st.write("Ce projet s’inscrit dans le domaine du **Machine Learning appliqué à l’analyse d’avis clients**.")

st.write("Notre objectif est de concevoir un **modèle de classification automatique** des avis publiés sur Trustpilot, afin d’en extraire des informations utiles.")

st.write("""
Concrètement, nous cherchons à identifier automatiquement les **thèmes évoqués dans chaque avis** (ex: service client, livraison, qualité des produits) pour renforcer la transparence et la lisibilité des notes d’entreprises.
""")

st.write("""
Cette catégorisation aiderait les utilisateurs à comprendre dans quels domaines une entreprise excelle ou rencontre des difficultés, et permettrait aux entreprises de cibler plus efficacement leurs axes d’amélioration.
""")

st.write("""
Le dataset fourni n'étant pas annoté pour entraîner un modèle sur cette tâche, notre premier objectif est d'obtenir un dataset entièrement labellisé avec des thématiques.
""")

st.write("""
Pour cela, nous avons utilisé le topic modeling pour identifier les principales thématiques et créer des clusters, que nous avons ensuite annotés manuellement.
""")

st.write("""
Les clusters homogènes ont été regroupés en trois thématiques : qualité produit, service livraison, service client.
""")

st.write("""
Enfin, nous avons utilisé les avis annotés à l'aide du topic modeling pour entraîner un modèle de classification multi-label dont l'objectif est d'annoter les avis provenant des clusters hétérogènes.
""")

st.header("Pipeline du projet")
st.write("""
Ce schéma présente les principales étapes de notre projet d'analyse des avis clients Trustpilot
         """)

st.image(
    "images/Trustpilot_Project.png",
    width=1000
)
# Another section
st.header("Groupe Projet")
st.write("Julie Boutelet - Laurine Charbonnier - Quentin Georges - Kinjal Kapadia")