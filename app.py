
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

# Another section
st.header("Groupe Projet")
st.write("Julie Boutelet - Laurine Charbonnier - Quentin Georges - Kinjal Kapadia")
