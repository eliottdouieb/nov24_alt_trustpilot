import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_score, recall_score
import plotly.express as px
import plotly.figure_factory as ff
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import torch
import scipy.special
st.set_page_config(
    page_title="Classification - Projet Trustpilot",
    layout="wide"
)

st.title("Classification des Avis")

# --- Section Configuration ---
st.markdown("### Configuration")

col1, col2 = st.columns(2)

MODEL_OPTIONS = [
    "SVM",
    "KNN",
    "Logistic Regression",
    "XGBoost",
    "CamemBERT"
]

DATA_OPTIONS = {
    "Topic Modelling": "bertopic",
    "Regex": "regex"
}

with col1:
    selected_model_name = st.selectbox("Choisir le modèle", MODEL_OPTIONS)

with col2:
    selected_data_type_label = st.radio("Données d'entraînement", list(DATA_OPTIONS.keys()))

selected_data_suffix = DATA_OPTIONS[selected_data_type_label]

# --- Chemins ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

def get_model_path(model_name, data_suffix):
    """Construit le chemin d'accès au fichier du modèle."""
    is_regex = data_suffix == "regex"

    if model_name == "Logistic Regression":
        if is_regex:
            return MODELS_DIR / "classification/logistic_regression/logistic_regression_regex.joblib"
        else:
            return MODELS_DIR / "classification/logistic_regression/logistic_regression.joblib"
    elif model_name == "XGBoost":
        if is_regex:
            return MODELS_DIR / "classification/xgboost/xgb_regex.pkl"
        else:
            return MODELS_DIR / "classification/xgboost/xgb_model.pkl"
    elif model_name == "SVM":
        if is_regex:
            return MODELS_DIR / "classification/svm/svm_regex.pkl"
        else:
            return MODELS_DIR / "classification/svm/svm_model.pkl"
    elif model_name == "KNN":
        if is_regex:
            return MODELS_DIR / "classification/knn/knn_regex.pkl"
        else:
            return MODELS_DIR / "classification/knn/knn_model.pkl"
    elif model_name == "CamemBERT":
        if is_regex:
            return MODELS_DIR / "classification/camembert/camembert_regex"
        else:
            return MODELS_DIR / "classification/camembert/camembert_valdataset"
    return None

# --- Fonctions Backend ---

@st.cache_resource
def load_embedding_model(model_name):
    """Charge efficacement le modèle Sentence Transformer."""
    if model_name == "Logistic Regression":
        return SentenceTransformer("dangvantuan/french-document-embedding", trust_remote_code=True)
    elif model_name == "XGBoost":
        return SentenceTransformer("camembert-base", trust_remote_code=True)
    return None

@st.cache_resource
def load_model(path, model_name):
    """Charge le modèle entraîné."""
    if isinstance(path, str):
        path = Path(path)
        
    if not path or not path.exists():
        return None
    
    if model_name == "CamemBERT":
        try:
            model = CamembertForSequenceClassification.from_pretrained(path)
            tokenizer = CamembertTokenizer.from_pretrained(path)
            return (model, tokenizer)
        except Exception as e:
            st.error(f"Erreur lors du chargement de CamemBERT: {e}")
            return None
    else:
        try:
            return joblib.load(path)
        except Exception as e:
             st.error(f"Erreur lors du chargement de {model_name}: {e}")
             return None

@st.cache_resource
def load_tfidf_model(model_name, data_suffix):
    """Charge le TfidfVectorizer pour SVM et KNN."""
    if model_name not in ["SVM", "KNN"]:
        return None
    
    is_regex = data_suffix == "regex"
    
    if model_name == "SVM":
        path = MODELS_DIR / ("classification/svm/tfidf_svm_regex.pkl" if is_regex else "classification/svm/tfidf_svm.pkl")
    else:
        path = MODELS_DIR / ("classification/knn/tfidf_knn_regex.pkl" if is_regex else "classification/knn/tfidf_knn.pkl")
        
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Erreur lors du chargement du vectoriseur TF-IDF : {e}")
            return None
    return None

def load_data(source_type):
    """Charge le jeu de données demandé."""
    if source_type == "100 avis annotés (Test)":
        path = DATA_DIR / "test_dataset/100_avis_annote.csv"
        if path.exists():
            return pd.read_csv(path, sep=";")
        return None
        
    elif source_type == "Avis non vus":
        path = DATA_DIR / "labelled_topics/dataset_avis.csv"
        if path.exists():
            df = pd.read_csv(path)
            return df.sample(n=50, random_state=42)
        return None
        
    return None

def predict_camembert(model_tuple, texts):
    """Effectue la prédiction avec CamemBERT."""
    model, tokenizer = model_tuple
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = torch.sigmoid(logits)
    
    # Seuil à 0.5
    preds = (probs > 0.5).int().numpy()
    return preds, probs.numpy()

# --- Logique Principale ---

# L'Embedder/Vectoriseur est nécessaire pour tous SAUF CamemBERT
if selected_model_name in ["SVM", "KNN"]:
    embedder = None
    with st.spinner("Chargement du vectoriseur TF-IDF..."):
         tfidf_vectorizer = load_tfidf_model(selected_model_name, selected_data_suffix)
elif selected_model_name in ["Logistic Regression", "XGBoost"]:
    tfidf_vectorizer = None
    with st.spinner("Chargement du modèle d'embedding..."):
        embedder = load_embedding_model(selected_model_name)
else:
    embedder = None
    tfidf_vectorizer = None

model_path = get_model_path(selected_model_name, selected_data_suffix)
clf = load_model(str(model_path) if model_path else "", selected_model_name)

if clf is None:
    st.error(f"Modèle introuvable : `{model_path}`")
    st.info("💡 Vérifiez que vous avez bien exécuté les notebooks d'entraînement pour générer ce modèle.")
elif selected_model_name in ["SVM", "KNN"] and tfidf_vectorizer is None:
    st.error("Vectoriseur TF-IDF introuvable pour ce modèle.")
    clf = None
else:
    st.success(f"Modèle chargé : {selected_model_name} ({selected_data_type_label})")

# selection du résultat
st.markdown("### Choix de l'évaluation")
result_mode = st.selectbox(
    "Quel résultat voulez-vous afficher ?", 
    ["Phrase personnalisée", "Arène des modèles", "100 avis annotés (Test)", "Avis non vus"]
)

labels = ["qualité produit", "service livraison", "service client"]

if result_mode == "Phrase personnalisée":
    st.subheader("Test en direct")
    user_input = st.text_area("Entrez un commentaire client :", "Ce produit est génial mais la livraison était catastrophique.")
    
    if st.button("Prédire") and clf is not None:
        if selected_model_name == "CamemBERT":
            prediction, proba = predict_camembert(clf, [user_input])
            prediction = prediction[0]
            proba = proba[0]
        else:
            if selected_model_name in ["KNN", "SVM"] and tfidf_vectorizer is not None:
                embedding = tfidf_vectorizer.transform([user_input])
            elif selected_model_name == "Logistic Regression":
                embedding = embedder.encode([user_input], normalize_embeddings=True)
            else:
                embedding = embedder.encode([user_input])
                
            prediction = clf.predict(embedding)[0]
            
            # Extraction personnalisée des probabilités pour contourner les incompatibilités de version scikit-learn
            try:
                # Une conversion dense peut être nécessaire pour la fonction de décision SVM/KNN
                if hasattr(clf, "estimators_") and hasattr(clf.estimators_[0], "decision_function"):
                    emb_for_decision = embedding.toarray() if hasattr(embedding, "toarray") else embedding
                    raw_decisions = np.array([e.decision_function(emb_for_decision) for e in clf.estimators_])
                    proba = scipy.special.expit(raw_decisions).ravel()
                elif hasattr(clf, "predict_proba"):
                    raw_proba = clf.predict_proba(embedding)
                    if isinstance(raw_proba, list):
                        proba = [p[0, 1] if p.shape[1] > 1 else p[0, 0] for p in raw_proba]
                    else:
                        proba = raw_proba[0]
                else:
                    proba = None
            except Exception as e:
                proba = None

        st.markdown(f"#### Résultat de la prédiction ({selected_model_name} - {selected_data_type_label})")
        cols = st.columns(len(labels))
        for i, label in enumerate(labels):
            with cols[i]:
                is_active = prediction[i] == 1
                color = "green" if is_active else "grey"
                st.markdown(f"**{label.capitalize()}**")
                if is_active:
                    st.success("OUI")
                else:
                    st.error("NON")
                
                if proba is not None:
                    st.progress(float(proba[i]))
                    st.caption(f"Confiance: {proba[i]:.2%}")

elif result_mode == "Arène des modèles":
    st.subheader("Arène des modèles 🥊")
    st.markdown("Testez tous les modèles simultanément sur un seul avis pour comparer leur robustesse.")
    
    user_input = st.text_area("Entrez un commentaire client :", "Ce produit est génial mais la livraison était catastrophique.")
    
    use_ground_truth = st.checkbox("Comparer avec les vraies étiquettes (Mode Correction)", value=False)
    true_labels = []
    if use_ground_truth:
        true_labels = st.multiselect(
            "Quelles sont les vraies étiquettes de cet avis ?",
            labels,
            default=[]
        )
    
    if st.button("Faire s'affronter les modèles 🥊"):
        if not user_input.strip():
            st.warning("Veuillez entrer un commentaire.")
        else:
            arena_results = []
            progress_bar = st.progress(0)
            
            for idx, m_name in enumerate(MODEL_OPTIONS):
                m_path = get_model_path(m_name, selected_data_suffix)
                t_model = load_model(str(m_path) if m_path else "", m_name)
                
                if t_model:
                    # Prediction logic
                    if m_name == "CamemBERT":
                        try:
                            preds, _ = predict_camembert(t_model, [user_input])
                            preds = preds[0]
                        except Exception as e:
                            st.warning(f"Erreur avec CamemBERT : {e}")
                            preds = [0, 0, 0]
                    elif m_name in ["KNN", "SVM"]:
                        t_tfidf = load_tfidf_model(m_name, selected_data_suffix)
                        if t_tfidf is not None:
                            emb_sparse = t_tfidf.transform([user_input])
                            preds = t_model.predict(emb_sparse)[0]
                        else:
                            preds = [0, 0, 0]
                    elif m_name == "Logistic Regression":
                        t_embedder = load_embedding_model(m_name)
                        if t_embedder is not None:
                            embeddings = t_embedder.encode([user_input], normalize_embeddings=True)
                            preds = t_model.predict(embeddings)[0]
                        else:
                            preds = [0, 0, 0]
                    else:
                        t_embedder = load_embedding_model(m_name)
                        if t_embedder is not None:
                            embeddings = t_embedder.encode([user_input])
                            preds = t_model.predict(embeddings)[0]
                        else:
                            preds = [0, 0, 0]
                    
                    row = {"Modèle": m_name}
                    for i, label in enumerate(labels):
                        pred_val = preds[i]
                        
                        if use_ground_truth:
                            is_true = label in true_labels
                            is_pred = bool(pred_val)
                            
                            if is_true == is_pred:
                                row[label.capitalize()] = "✅ OUI" if is_pred else "✅ NON"
                            else:
                                row[label.capitalize()] = "❌ OUI" if is_pred else "❌ NON"
                        else:
                            row[label.capitalize()] = "OUI" if pred_val else "NON"
                    
                    arena_results.append(row)
                
                progress_bar.progress((idx + 1) / len(MODEL_OPTIONS))
            
            progress_bar.empty()
            
            if arena_results:
                st.markdown(f"### Résultats de l'Arène ({selected_data_type_label})")
                
                res_df = pd.DataFrame(arena_results)
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                
                st.caption("💡 **Note** : Ce test unitaire a pour but de valider le comportement des modèles sur des cas aux limites (sarcasme, ambiguïté). "
                           "Pour juger de la véritable performance statistique globale, l'onglet '100 avis annotés (Test)' reste le seul juge de paix.")
            else:
                st.warning("Aucun modèle n'a pu être chargé.")

elif result_mode in ["100 avis annotés (Test)", "Avis non vus"]:
    st.subheader(f"Analyse sur : {result_mode}")
    
    df = load_data(result_mode)
    
    if df is None:
        st.error("Fichier de données introuvable.")
    else:
        if clf is None:
            st.warning("Veuillez charger un modèle valide pour voir les prédictions.")
        else:
            text_col = "clean_comment" if "clean_comment" in df.columns else "Commentaire"
            
            if text_col not in df.columns:
                st.error(f"Colonne de texte '{text_col}' absente du fichier.")
            else:
                texts = df[text_col].astype(str).tolist()
                
                if selected_model_name == "CamemBERT":
                    with st.spinner(f"Prédiction CamemBERT sur {len(texts)} textes..."):
                        predictions, _ = predict_camembert(clf, texts)
                else:
                    with st.spinner(f"Encodage de {len(texts)} textes..."):
                        if selected_model_name in ["KNN", "SVM"] and tfidf_vectorizer is not None:
                            embeddings = tfidf_vectorizer.transform(texts)
                        elif selected_model_name == "Logistic Regression":
                            embeddings = embedder.encode(texts, normalize_embeddings=True)
                        else:
                            embeddings = embedder.encode(texts)
                    predictions = clf.predict(embeddings)
                    
                    try:
                        if hasattr(clf, "estimators_") and hasattr(clf.estimators_[0], "decision_function"):
                            emb_for_decision = embeddings.toarray() if hasattr(embeddings, "toarray") else embeddings
                            raw_decisions = np.array([e.decision_function(emb_for_decision) for e in clf.estimators_])
                            probas_table = scipy.special.expit(raw_decisions).T
                        elif hasattr(clf, "predict_proba"):
                            raw_proba = clf.predict_proba(embeddings)
                            if isinstance(raw_proba, list):
                                probas_table = np.array([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in raw_proba]).T
                            else:
                                probas_table = raw_proba
                        else:
                            probas_table = None
                    except Exception:
                         probas_table = None
                
                res_df = df.copy()
                for i, label in enumerate(labels):
                    res_df[f"Pred_{label}"] = predictions[:, i]
                    if selected_model_name != "CamemBERT" and probas_table is not None:
                        res_df[f"Prob_{label}"] = probas_table[:, i]

                disp_cols = [text_col] + [f"Pred_{l}" for l in labels]
                if selected_model_name != "CamemBERT" and probas_table is not None:
                    disp_cols += [f"Prob_{l}" for l in labels]
                st.dataframe(res_df[disp_cols])
                
                # Pour "100 avis annotés", on affiche les métriques
                if result_mode == "100 avis annotés (Test)":
                    try:
                        y_true = df[labels].values
                        
                        st.markdown("### Métriques de Performance")
                        report = classification_report(y_true, predictions, target_names=labels, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
                        
                        # Matrice de confusion pour chaque label
                        st.markdown("### Matrices de Confusion")
                        cm_cols = st.columns(3)
                        for i, label in enumerate(labels):
                            with cm_cols[i]:
                                st.write(f"**{label}**")
                                cm = confusion_matrix(y_true[:, i], predictions[:, i])
                                fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                                labels=dict(x="Prédit", y="Réel", color="Nombre"),
                                                x=['Non', 'Oui'], y=['Non', 'Oui'])
                                st.plotly_chart(fig, use_container_width=True)
                                
                    except KeyError as e:
                        st.error(f"Impossible de calculer les métriques : colonnes cibles manquantes ({e})")

# --- Grille de comparaison ---
st.markdown("---")
st.header("Comparateur de Modèles")

if st.checkbox("Activer la comparaison (Peut être lent)"):
    if result_mode != "100 avis annotés (Test)":
        st.warning("La comparaison est disponible uniquement sur les données annotées (Test) pour avoir des mesures fiables.")
    else:
        df_test = load_data("100 avis annotés (Test)")
        if df_test is not None:
            texts = df_test["clean_comment"].astype(str).tolist()
            y_true = df_test[labels].values
            
            comparison_results = []
            progress_bar = st.progress(0)
            
            # Logique Embeddings vs TF-IDF
            embedder_cache = {}
            for idx, m_name in enumerate(MODEL_OPTIONS):
                m_path = get_model_path(m_name, selected_data_suffix)
                t_model = load_model(str(m_path) if m_path else "", m_name)
                
                if t_model:
                    if m_name == "CamemBERT":
                        preds, _ = predict_camembert(t_model, texts)
                    elif m_name in ["KNN", "SVM"]:
                        t_tfidf = load_tfidf_model(m_name, selected_data_suffix)
                        if t_tfidf is not None:
                            emb_sparse = t_tfidf.transform(texts)
                            preds = t_model.predict(emb_sparse)
                        else:
                            continue # ignorer si tfidf est manquant
                    else:
                        embedder = load_embedding_model(m_name)
                        if embedder is not None:
                            if m_name not in embedder_cache:
                                if m_name == "Logistic Regression":
                                    embedder_cache[m_name] = embedder.encode(texts, normalize_embeddings=True)
                                else:
                                    embedder_cache[m_name] = embedder.encode(texts)
                            embeddings = embedder_cache[m_name]
                            preds = t_model.predict(embeddings)
                        else:
                            continue
                    f1_macro = f1_score(y_true, preds, average='macro', zero_division=0)
                    prec_macro = precision_score(y_true, preds, average='macro', zero_division=0)
                    rec_macro = recall_score(y_true, preds, average='macro', zero_division=0)
                    
                    comparison_results.append({
                        "Modèle": m_name,
                        "F1-Score (Macro)": f1_macro,
                        "Précision (Macro)": prec_macro,
                        "Rappel (Macro)": rec_macro
                    })
                
                progress_bar.progress((idx + 1) / len(MODEL_OPTIONS))
            
            if comparison_results:
                st.markdown(f"### Comparatif ({selected_data_type_label})")
                comp_df = pd.DataFrame(comparison_results).sort_values(by="F1-Score (Macro)", ascending=False)
                st.dataframe(comp_df.style.background_gradient(cmap='Blues', subset=["F1-Score (Macro)", "Précision (Macro)", "Rappel (Macro)"]))
                
                fig = px.bar(comp_df, x="Modèle", y="F1-Score (Macro)", color="F1-Score (Macro)", range_y=[0, 1])
                st.plotly_chart(fig)
            else:
                st.warning("Aucun modèle n'a pu être chargé pour la comparaison.")
