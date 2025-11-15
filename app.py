# ===============================================================
# 🚀 App.py — Point d'entrée principal de l'application
# ===============================================================

import streamlit as st
import os

# Configuration de base
st.set_page_config(
    page_title="Modélisation des Ventes Publicitaires",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"  # ← Ajout de cette ligne pour masquer la sidebar par défaut
)

# Masquer complètement la sidebar
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Redirection automatique vers la page d'accueil
accueil_path = os.path.join("pages", "Accueil.py")

# Exécution automatique du fichier d'accueil
with open(accueil_path, "r", encoding="utf-8") as f:
    exec(f.read())