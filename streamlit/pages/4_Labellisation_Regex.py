import streamlit as st
import pandas as pd
from pathlib import Path
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Labellisation Regex - Projet Trustpilot",
    layout="wide"
)

st.title("Labellisation par Expressions Régulières (Regex)")

st.markdown("""
Cette page présente une approche alternative à notre méthode principale de **Topic Modelling**.
     
Face à la complexité d'obtenir des clusters parfaitement séparés avec le Topic Modelling, nous avons développé cette méthode complémentaire utilisant des **expressions régulières (Regex)**. 
Contrairement aux modèles de Machine Learning, cette approche est **100% transparente et interprétable**, car elle repose sur des règles métiers simples définies manuellement pour cibler les plaintes ou compliments les plus fréquents.
""")

# --- Chemins ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# --- Dictionnaires ---
qualite_keywords = [
    r"cass(é|ée|és|ées)?",
    r"casse(r)?",
    r"qualit(é|e|és|es|ées|ees)?(?!\s+de\s+(service|livraison|sav))",
    r"mati[èe]re(s)?",
    r"soli(de|d|de)?",
    r"résistan(t|ce|ces|t)?",
    r"correspond(ait|s|t)?",
    r"fonctionne(r)?",
    r"d[ée]f(ec)?tueu(se|s|e|x|ses)?",
    r"tiss(u|us|u)?(x)?",
    r"couleur(s|e)?",
    r"((trop|un peu|très)\s+(petit|grand)(e|es)?|taille(nt)?\s+(trop|un peu|très)?\s*(petit|grand)(e|es)?)",
    r"grand(e|es|s)?",
    r"ray(é|ee|ée|e|és|és|ees|ée|ure|ures)",
    r"finitions?",
    r"beau(x)?",
    r"belle(s)?",
    r"joli(e|es|s)?",
    r"conforme(s)?",
    r"en panne(s)?",
    r"bas de gamme",
    r"inutilisable(s)?"
]

livraison_keywords = [
    r"livr(é|er|aison|aisons|ées|ée|eur)?",
    r"passage(s)?",
    r"coli(s)?",
    r"paquet(s)?",
    r"carton(s)?",
    r"retard(é|ée|s)?",
    r"dpd",
    r"gls",
    r"poste",
    r"facteur(s)?",
    r"mondial relay",
    r"mondial relais",
    r"relais",
    r"relay",
    r"point relais",
    r"domicile",
    r"retrait",
    r"bureau de poste",
    r"ouvert(s|e|es)?",
    r"endommag(é|ée|és|ées)?",
    r"déposé(e|s)?",
    r"laisser?|laissé(e|s)?",
    r"transporteur(s)?",
    r"manquant(e|s)?",
    r"incomplet(s|es)?",
    r"abîmé(e|s)?",
    r"déchiré(e|s)?",
]

client_keywords = [
    r"service client(s)?",
    r"\bsav\b",
    r"service après vente",
    r"répond(re|s|u)?",
    r"réponse(s)?",
    r"rembour(s|ser|sement|sez)?",
    r"contact(er|e|es)?",
    r"mail(s)?|email(s)?",
    r"appel(er|s)?",
    r"incompétent(e|s|es)?",
    r"incompétence",
    r"escroc(s)?",
    r"arnaque(s)?",
    r"voleur(s|ses)?",
    r"solution(s)?",
    r"aucune réponse",
    r"réclamation(s)?",
    r"annul(é|ée|er|ation|ations)",
    r"litige(s)?",
    r"impossible[^.]{0,40}(joindre|contact(er|é|és)?|contacter)",
    r"qualité\s+de\s+(service|sav|service client)",
]

def make_regex(words):
    return r"\b(" + "|".join(words) + r")\b"

regex_dict = {
    "qualité produit": make_regex(qualite_keywords),
    "service livraison": make_regex(livraison_keywords),
    "service client": make_regex(client_keywords)
}

# --- 1. Explication de la Méthodologie ---
st.header("1. Méthodologie et Dictionnaires")
st.markdown("Nous avons défini 3 catégories principales en fonction de l'analyse lexicale. Cliquez pour voir les expressions régulières (Regex) associées à chaque label :")

col1, col2, col3 = st.columns(3)
with col1:
    with st.expander("🛍️ Qualité Produit"):
         st.code("\n".join(qualite_keywords), language="regex")
with col2:
    with st.expander("📦 Service Livraison"):
         st.code("\n".join(livraison_keywords), language="regex")
with col3:
    with st.expander("🎧 Service Client"):
         st.code("\n".join(client_keywords), language="regex")

# --- Processing des KPIs ---
@st.cache_data
def process_regex_data():
    path = DATA_DIR / "dataset_final.pkl"
    if not path.exists():
        return None
    
    df = pd.read_pickle(path)
    if "comment_id" not in df.columns:
        df["comment_id"] = df.index
    
    # Appliquer les regex avec case=False (insensible à la casse)
    for label, pattern in regex_dict.items():
        df[label] = df["clean_comment"].str.contains(pattern, case=False, na=False).astype(int)
        
    df["nb_labels"] = df[["qualité produit", "service livraison", "service client"]].sum(axis=1)
    df["is_annotated"] = df["nb_labels"] > 0
    return df

with st.spinner("Recherche des motifs Regex sur les 100 000+ avis (mis en cache)..."):
    df_regex = process_regex_data()

if df_regex is not None:
    # --- 2. Statistiques Globales ---
    st.header("2. Statistiques des Annotations")
    
    total_avis = len(df_regex)
    annotated = df_regex["is_annotated"].sum()
    coverage = (annotated / total_avis) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total des avis traités", f"{total_avis:,}".replace(",", " "))
    c2.metric("Avis annotés par Regex", f"{annotated:,}".replace(",", " "))
    c3.metric("Avis non classés (Restants)", f"{total_avis - annotated:,}".replace(",", " "))
    c4.metric("Taux de couverture du dataset", f"{coverage:.1f}%")

    st.markdown("### Répartition des Catégories Détectées")
    
    df_annotated = df_regex[df_regex["is_annotated"]]
    label_counts = df_annotated[["qualité produit", "service livraison", "service client"]].sum().reset_index()
    label_counts.columns = ["Catégorie", "Nombre d'avis"]
    label_counts["Catégorie"] = label_counts["Catégorie"].str.title()
    
    col_g1, col_g2 = st.columns((3, 2))
    with col_g1:
        fig_bar = px.bar(
            label_counts, 
            x="Nombre d'avis", 
            y="Catégorie", 
            orientation='h',
            text_auto=True,
            title="Volume d'avis par catégorie trouvée",
            color="Catégorie",
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
        )
        fig_bar.update_layout(showlegend=False, yaxis_title=None)
        st.plotly_chart(fig_bar, use_container_width=True)
         
    with col_g2:
        multi_label_counts = df_annotated["nb_labels"].value_counts().reset_index()
        multi_label_counts.columns = ["Nombre de thèmes", "Quantité"]
        multi_label_counts["Nombre de thèmes"] = multi_label_counts["Nombre de thèmes"].astype(str) + " thème(s)"
        
        fig_pie = px.pie(
            multi_label_counts, 
            values="Quantité", 
            names="Nombre de thèmes", 
            title="Mixité des avis (Combinaison de Labels)",
            hole=0.4, 
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- 3. Explorateur ---
    st.header("3. Explorateur de Données")
    st.markdown("Afin de vérifier la pertinence de cette méthode manuelle, vous pouvez consulter un échantillon d'avis pour la catégorie de votre choix.")
    
    selected_label = st.selectbox(
        "Filtrer les commentaires par catégorie détectée :", 
        ["Toutes les catégories détectées", "qualité produit", "service livraison", "service client", "Aucun label détecté"]
    )
    
    if selected_label == "Toutes les catégories détectées":
        df_sample = df_annotated
    elif selected_label == "Aucun label détecté":
        df_sample = df_regex[~df_regex["is_annotated"]]
    else:
        df_sample = df_annotated[df_annotated[selected_label] == 1]
    
    if len(df_sample) > 0:
        # On renomme pour que ce soit plus joli à lire
        display_df = df_sample[["clean_comment", "qualité produit", "service livraison", "service client"]].sample(min(100, len(df_sample)))
        display_df.rename(columns={"clean_comment": "Commentaire Nettoyé"}, inplace=True)
        # Convert numeric 0/1 to bool for nice checkmarks in streamlit dataframe
        display_df["qualité produit"] = display_df["qualité produit"].astype(bool)
        display_df["service livraison"] = display_df["service livraison"].astype(bool)
        display_df["service client"] = display_df["service client"].astype(bool)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption("Affiche jusqu'à 100 avis tirés au hasard parmi les avis correspondants.")
    else:
        st.info("Aucun avis trouvé pour cette sélection.")

else:
    st.error(f"Fichier de données introuvable ({DATA_DIR / 'dataset_final.pkl'}).")

# --- 4. Playground ---
st.markdown("---")
st.header("4. Playground (Test Interactif)")
st.markdown("Rédigez un commentaire de test et vérifiez les labels détectés !")

test_text = st.text_area(
    "Commentaire de test :", 
    "Ce produit est magnifique, vraiment beau. Mais le colis est arrivé complètement déchiré et le SAV est incompétent, aucune réponse de leur part !"
)

if test_text:
    st.markdown("#### Labels détectés par nos règles :")
    res_cols = st.columns(3)
    
    for i, (label, pattern) in enumerate(regex_dict.items()):
        # Utiliser re.search pour détecter une correspondance (insensible à la casse)
        match = re.search(pattern, test_text, re.IGNORECASE)
        
        with res_cols[i]:
            st.markdown(f"**{label.title()}**")
            if match:
                st.success(f"✅ DÉTECTÉ")
                st.caption(f"Mot-clé déclencheur : **`{match.group()}`**")
            else:
                st.error("❌ NON DÉTECTÉ")
