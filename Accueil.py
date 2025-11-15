# Accueil.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Analyse des Ventes Publicitaires",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec animations améliorées
st.markdown("""
<style>
    /* Styles généraux - COMMENCER PLUS HAUT */
    .main {
        padding-top: 0rem !important;
    }
    
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem !important;
        margin-top: 0rem !important;
        font-weight: bold;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 2s ease-in;
    }
    
    /* Styles améliorés pour la sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 50%, #1a252f 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 50%, #1a252f 100%) !important;
        padding: 1rem 0.8rem !important;
        box-shadow: 4px 0 15px rgba(0,0,0,0.1);
    }
    
    .sidebar-header {
        text-align: center;
        margin-bottom: 1rem !important;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid rgba(255,255,255,0.1);
    }
    
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.3rem;
        background: linear-gradient(45deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sidebar-subtitle {
        font-size: 0.85rem;
        color: #bdc3c7;
        opacity: 0.8;
    }
    
    /* Navigation améliorée - ESPACES RÉDUITS AU MAXIMUM */
    .stRadio > div {
        flex-direction: column;
        gap: 0.05rem !important;
    }
    
    .stRadio > div > label {
        background: rgba(255,255,255,0.05);
        padding: 8px 10px !important;
        border-radius: 6px;
        margin: 0.5px 0 !important;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        min-height: auto !important;
        height: auto !important;
    }
    
    .stRadio > div > label:hover {
        background: rgba(255,255,255,0.1);
        transform: translateX(3px);
        border-color: rgba(52, 152, 219, 0.5);
    }
    
    .stRadio > div > label[data-testid="stRadio"] > div:first-child {
        color: #ecf0f1 !important;
        font-weight: 500;
        padding: 0 !important;
        font-size: 0.9rem;
    }
    
    /* Réduction des espaces dans les sections */
    .upload-section {
        background: rgba(255,255,255,0.05);
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.8rem 0 !important;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .upload-header {
        color: #3498db;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Bouton de téléchargement avec espace réduit */
    .sidebar-download-btn {
        background: linear-gradient(45deg, #27ae60, #2ecc71);
        color: white;
        border: none;
        padding: 10px 14px !important;
        border-radius: 8px;
        font-weight: 600;
        font-size: 12px;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        animation: pulse 2s infinite;
        text-decoration: none;
        text-align: center;
        width: 100%;
        margin: 6px 0 !important;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .sidebar-download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(39, 174, 96, 0.6);
        background: linear-gradient(45deg, #229954, #27ae60);
        color: white;
        text-decoration: none;
    }
    
    /* Info box avec espace réduit */
    .info-box {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.2), rgba(155, 89, 182, 0.2));
        border: 1px solid rgba(255,255,255,0.1);
        padding: 0.7rem;
        border-radius: 6px;
        margin: 0.6rem 0 !important;
        color: #ecf0f1;
        font-size: 0.75rem;
        line-height: 1.2;
    }
    
    .info-box strong {
        color: #3498db;
    }
    
    /* Status indicators avec espace réduit */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 0.2rem 0 !important;
    }
    
    .status-ready {
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    
    .status-waiting {
        background: rgba(241, 196, 15, 0.2);
        color: #f1c40f;
        border: 1px solid rgba(241, 196, 15, 0.3);
    }
    
    /* Réduction des marges pour tous les éléments de la sidebar */
    .sidebar .element-container {
        margin-bottom: 0.3rem !important;
    }
    
    .sidebar .stRadio {
        margin-bottom: 0.3rem !important;
    }
    
    .sidebar .stFileUploader {
        margin-bottom: 0.3rem !important;
    }
    
    .sidebar .stExpander {
        margin-bottom: 0.3rem !important;
    }
    
    /* Réduction des espaces dans le contenu principal */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from { transform: translateY(50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4); }
        50% { transform: scale(1.02); box-shadow: 0 6px 20px rgba(39, 174, 96, 0.6); }
        100% { transform: scale(1); box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4); }
    }
    
    /* Styles pour les autres éléments */
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        color: white;
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
        animation: fadeInUp 1s ease-out;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.8rem;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        color: white;
        border: none;
        padding: 0.4rem 1.5rem;
        border-radius: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de la session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def main():
    # CSS global professionnel
    st.markdown("""
    <style>
    .professional-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem !important;
        margin-top: 0.5rem !important;
        font-weight: 700;
        animation: fadeInScale 1.2s ease-in-out;
    }
    
    .author-signature {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 18px;
        font-weight: 600;
        text-align: center;
        margin: 0 auto 1.5rem auto;
        width: fit-content;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    
    .dynamic-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .dynamic-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }
    
    /* Réduction des marges globales */
    .css-1v0mbdj, .css-1v3fvcr, .css-1r6slb0 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar - Navigation avec espaces réduits
    with st.sidebar:
        # Bouton de téléchargement
        download_url = "https://drive.google.com/uc?export=download&id=1aaA67Bci5UJaWSDC-iaVAEnqwSIq3q3C"
        
        st.markdown(f"""
        <a href="{download_url}" download="base_donnees_test.xlsx" class="sidebar-download-btn">
            📥 Télécharger la Base de Test
        </a>
        """, unsafe_allow_html=True)
        
        # Info box améliorée
        st.markdown("""
        <div class="info-box">
            <strong>💡 Base de test prête à utiliser</strong><br>
            Téléchargez et importez cette base pour tester toutes les fonctionnalités de l'application avec des données optimisées.
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation entre les pages - ESPACES RÉDUITS AU MAXIMUM
        st.markdown("#### 🧭 Navigation")
        page = st.radio(
            " ",
            ["🏠 Accueil", "📈 Analyse Exploratoire", "💾 Données", "🤖 Performances", "🔮 Prédiction"],
            index=0,
            label_visibility="collapsed"
        )
        
        # Section d'upload améliorée
        st.markdown("""
        <div class="upload-section">
            <div class="upload-header">
                📁 Importation des Données
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choisissez votre fichier Excel",
            type=['xlsx', 'xls'],
            help="Format attendu : colonnes TV, Radio, Newspaper, Sales (4 variables avec Sales comme variable cible)"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("✅ Données chargées avec succès!")
                
                # Aperçu des données dans un expander stylisé
                with st.expander("👀 Aperçu des données chargées", expanded=False):
                    st.dataframe(df.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement: {str(e)}")
        else:
            st.info("📝 Veuillez importer un fichier Excel pour commencer l'analyse")

    # Contenu principal basé sur la page sélectionnée
    if page == "🏠 Accueil":
        show_home_page()
    elif page == "📈 Analyse Exploratoire":
        if st.session_state.data_loaded:
            import Analyse_Exploratoire
            Analyse_Exploratoire.main()
        else:
            st.warning("⚠️ Veuillez d'abord importer des données dans l'onglet Accueil")
    elif page == "💾 Données":
        if st.session_state.data_loaded:
            import Donnees
            Donnees.main()
        else:
            st.warning("⚠️ Veuillez d'abord importer des données dans l'onglet Accueil")
    elif page == "🤖 Performances":
        if st.session_state.data_loaded:
            import Performence
            Performence.main()
        else:
            st.warning("⚠️ Veuillez d'abord importer des données dans l'onglet Accueil")
    elif page == "🔮 Prédiction":
        if st.session_state.data_loaded:
            import Prediction
            Prediction.main()
        else:
            st.warning("⚠️ Veuillez d'abord importer des données dans l'onglet Accueil")

def show_home_page():
    """Affiche la page d'accueil"""
    
    # Introduction - COMMENCER PLUS HAUT
    st.markdown(
    """
    <div style='text-align: left; font-weight: bold; font-size: 16px; margin-top: 0.5rem; margin-bottom: 0.5rem;'>
        👨‍💻 Par <span style='color:#1f77b4;'>Lassina SANOU</span>
    </div>
    """,
    unsafe_allow_html=True
    )
    
    st.markdown("""
    <div class="card">
        <h2>BIENVENUE SUR VOTRE PLATEFORME D'ANALYSE DE DONNEES ET DE MODELISATION PREDICTIVE DES VENTES PUBLICIAIRES</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>Cette plateforme est conçue pour vous offrir une expérience complète d'analyse des performances marketing à travers la data science. 
        Elle vous permet d'explorer, visualiser et interpréter en profondeur l'impact des différents canaux publicitaires 
        (télévision, radio, presse, digital, etc.) sur le chiffre d'affaires généré. 
        Grâce à des outils de modélisation statistique et de machine learning.</p>
        <p>L'objectif principal est de transformer vos données en insights actionnables pour soutenir la prise de décision stratégique. 
        Vous disposerez d'indicateurs dynamiques, de graphiques interactifs et de modèles prédictifs qui facilitent la compréhension 
        des tendances, la détection d'opportunités et l'amélioration continue des performances commerciales.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Information sur la base de test
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4ECDC4, #44A08D); padding: 1.2rem; border-radius: 12px; color: white; margin: 1.5rem 0;'>
        <h4 style='color: white; margin-bottom: 0.8rem;'>💡 Base de Données de Test Disponible</h4>
        <p style='margin-bottom: 0;'>Utilisez le bouton <strong>"📥 Télécharger la Base de Test"</strong> dans la barre latérale pour télécharger directement notre base de données d'exemple. 
        Cette base contient des données formatées avec les variables TV, Radio, Newspaper et Sales, parfaitement adaptées pour tester 
        toutes les fonctionnalités de l'application !</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sections features
    st.markdown("## 🎯 Fonctionnalités Principales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="feature-icon">🔍</div>
            <h4>Analyse Exploratoire</h4>
            <p>Explorez vos données avec des visualisations interactives 
            et découvrez les tendances cachées.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="feature-icon">🤖</div>
            <h4>Modélisation IA</h4>
            <p>Comparez différents algorithmes de machine learning 
            et sélectionnez le meilleur modèle.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <div class="feature-icon">🔮</div>
            <h4>Prédictions</h4>
            <p>Obtenez des prédictions précises basées sur 
            vos budgets publicitaires.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Contexte du projet
    st.markdown("## 📋 Contexte du Projet")
    
    st.markdown(
        """
    <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 5px solid #1f77b4; margin: 1rem 0;'>
        <h4 style='color: #1f77b4;'>Objectifs Business</h4>
        <ul>
            <li>📊 Comprendre l'impact de chaque canal publicitaire sur les ventes</li>
            <li>🎯 Optimiser l'allocation du budget marketing</li>
            <li>🔮 Prédire les ventes futures avec précision</li>
            <li>💡 Fournir des insights actionnables pour les décisions business</li>
        </ul>
        <h4 style='color: #1f77b4; margin-top: 1.5rem;'>Variables Analysées</h4>
        <ul>
            <li><strong>TV Ad Budget ($)</strong> - Budget publicitaire télévision</li>
            <li><strong>Radio Ad Budget ($)</strong> - Budget publicitaire radio</li>
            <li><strong>Newspaper Ad Budget ($)</strong> - Budget publicitaire presse</li>
            <li><strong>Sales ($)</strong> - Chiffre d'affaires généré (variable cible)</li>
        </ul>
    </div>
    """, 
    unsafe_allow_html=True)
    
    # Guide de démarrage
    st.markdown("## 🚀 Guide de Démarrage Rapide")
    st.markdown("#### Ce guide vous accompagne pas à pas dans la prise en main de la plateforme d'analyse et de modélisation prédictive des ventes publicitaires.")

    steps = [
        {"icon": "📥", "title": "Télécharger la base de test", "desc": "Commencez par télécharger notre base de données d'exemple en cliquant sur le bouton '📥 Télécharger la Base de Test' dans la barre latérale. Le téléchargement démarre directement avec un fichier Excel prêt à l'emploi."},
        {"icon": "📁", "title": "Importer vos données", "desc": "Chargez votre fichier de données au format Excel (.xlsx) à partir du menu latéral. Le système détecte automatiquement les colonnes numériques et catégorielles, vérifie la qualité des données et gère les valeurs manquantes. Vous pouvez ensuite visualiser un aperçu du jeu de données pour confirmer que tout est correctement importé."},
        {"icon": "🔍", "title": "Explorer les données", "desc": "Dans cette phase, explorez les caractéristiques principales de vos données à travers des statistiques descriptives et des visualisations interactives. Analysez la distribution de chaque variable, identifiez les valeurs aberrantes et observez les corrélations entre les budgets publicitaires et les ventes. Ces analyses permettent de mieux comprendre la structure des données avant toute modélisation."},
        {"icon": "🤖", "title": "Évaluer les modèles", "desc": "Testez et comparez plusieurs modèles de régression (linéaire, Ridge, Lasso, Random Forest, etc.) afin d'évaluer leurs performances selon différents indicateurs tels que le R², la MAE ou la RMSE. La plateforme sélectionne automatiquement le modèle le plus performant et vous permet de visualiser les courbes de prédiction et les erreurs associées."},
        {"icon": "🔮", "title": "Faire des prédictions", "desc": "Une fois le meilleur modèle sélectionné, vous pouvez effectuer des prévisions en saisissant de nouvelles valeurs de budget publicitaire. Le modèle génère instantanément les ventes estimées, accompagnées d'un intervalle de confiance. Cette étape vous aide à simuler différents scénarios marketing et à orienter vos décisions stratégiques de manière data-driven."}
    ]
    
    for i, step in enumerate(steps, 1):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"<h3 style='text-align: center; color: #1f77b4;'>{step['icon']}</h3>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**Étape {i}: {step['title']}**")
            st.markdown(f"<small>{step['desc']}</small>", unsafe_allow_html=True)
        
        if i < len(steps):
            st.markdown("---")

if __name__ == "__main__":
    main()