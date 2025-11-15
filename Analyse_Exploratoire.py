# Analyse_Exploratoire.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def main():
    st.markdown("""
    <style>
    .exploration-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="exploration-header">🔍 Analyse Exploratoire des Données</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Veuillez d'abord importer des données dans la page d'Accueil")
        return
    
    df = st.session_state.df
    
    # Onglets pour l'analyse exploratoire
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Aperçu Général", 
        "📈 Distributions", 
        "🔗 Corrélations", 
        "🎯 Insights"
    ])
    
    with tab1:
        show_general_overview(df)
    
    with tab2:
        show_distributions(df)
    
    with tab3:
        show_correlations(df)
    
    with tab4:
        show_insights(df)

def show_general_overview(df):
    st.header("📊 Aperçu Général des Données")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre d'Observations", len(df))
    
    with col2:
        st.metric("Nombre de Variables", len(df.columns))
    
    with col3:
        st.metric("Données Manquantes", df.isnull().sum().sum())
    
    with col4:
        st.metric("Valeurs Dupliquées", df.duplicated().sum())
    
    # Statistiques descriptives
    st.subheader("📋 Statistiques Descriptives")
    st.dataframe(df.describe().round(2))
    
    # Aperçu des données
    st.subheader("👀 Aperçu des Données")
    st.dataframe(df.head(10))
    
    # Informations sur les types de données
    st.subheader("🔧 Types de Données")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Types de données:")
        dtype_df = pd.DataFrame(df.dtypes, columns=['Type'])
        st.dataframe(dtype_df)
    
    with col2:
        st.write("Valeurs manquantes par colonne:")
        missing_df = pd.DataFrame(df.isnull().sum(), columns=['Valeurs Manquantes'])
        st.dataframe(missing_df)

def show_distributions(df):
    st.header("📈 Analyse des Distributions")
    
    # Sélection de la variable à visualiser
    selected_var = st.selectbox(
        "Sélectionnez une variable pour l'analyse:",
        df.columns
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogramme avec Plotly
        fig_hist = px.histogram(
            df, 
            x=selected_var,
            nbins=30,
            title=f"Distribution de {selected_var}",
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(
            xaxis_title=selected_var,
            yaxis_title="Fréquence",
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot
        fig_box = px.box(
            df, 
            y=selected_var,
            title=f"Box Plot de {selected_var}",
            color_discrete_sequence=['#ff7f0e']
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Analyse de la normalité
    st.subheader("📊 Test de Normalité")
    
    if st.checkbox("Effectuer le test de normalité (Shapiro-Wilk)"):
        data = df[selected_var].dropna()
        stat, p_value = stats.shapiro(data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Statistique de test", f"{stat:.4f}")
        with col2:
            st.metric("P-value", f"{p_value:.4f}")
        
        if p_value > 0.05:
            st.success("✅ La variable suit une distribution normale (p > 0.05)")
        else:
            st.warning("❌ La variable ne suit pas une distribution normale (p ≤ 0.05)")
    
    # Distribution multivariée
    st.subheader("🎯 Distribution Multivariée")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Variable X:", df.columns, index=0)
    with col2:
        y_var = st.selectbox("Variable Y:", df.columns, index=3 if 'Sales' in df.columns else 0)
    
    fig_scatter = px.scatter(
        df,
        x=x_var,
        y=y_var,
        title=f"Relation entre {x_var} et {y_var}",
        trendline="ols",
        color_discrete_sequence=['#2ca02c']
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

def show_correlations(df):
    st.header("🔗 Analyse des Corrélations")
    
    # Matrice de corrélation
    corr_matrix = df.corr()
    
    # Heatmap avec Plotly
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        hoverongaps=False,
        text=corr_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig_heatmap.update_layout(
        title="Matrice de Corrélation",
        xaxis_title="Variables",
        yaxis_title="Variables",
        width=600,
        height=600
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Corrélations avec la variable cible
    if 'Sales' in df.columns:
        st.subheader("🎯 Corrélations avec les Ventes")
        
        sales_corr = df.corr()['Sales'].sort_values(ascending=False)
        sales_corr_df = pd.DataFrame({
            'Variable': sales_corr.index,
            'Corrélation': sales_corr.values
        }).round(3)
        
        # Graphique à barres des corrélations
        fig_bar = px.bar(
            sales_corr_df,
            x='Variable',
            y='Corrélation',
            title="Corrélations avec les Ventes",
            color='Corrélation',
            color_continuous_scale='RdYlBu_r',
            range_color=[-1, 1]
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Affichage du tableau
        st.dataframe(sales_corr_df)

def show_insights(df):
    st.header("🎯 Insights Business")
    
    if 'Sales' not in df.columns:
        st.warning("La variable 'Sales' est requise pour cette analyse")
        return
    
    # KPI principaux
    st.subheader("📊 Indicateurs Clés de Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = df['Sales'].sum()
        st.metric("💰 Ventes Totales", f"${total_sales:,.0f}")
    
    with col2:
        avg_sales = df['Sales'].mean()
        st.metric("📈 Ventes Moyennes", f"${avg_sales:.2f}")
    
    with col3:
        roi_tv = df['Sales'].sum() / df['TV_Ad_Budget'].sum() if 'TV_Ad_Budget' in df.columns else 0
        st.metric("📺 ROI TV", f"{roi_tv:.2f}x")
    
    with col4:
        roi_radio = df['Sales'].sum() / df['Radio_Ad_Budget'].sum() if 'Radio_Ad_Budget' in df.columns else 0
        st.metric("📻 ROI Radio", f"{roi_radio:.2f}x")
    
    # Analyse d'efficacité des canaux
    st.subheader("📊 Efficacité des Canaux Publicitaires")
    
    if all(col in df.columns for col in ['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget']):
        # Budget total par canal
        budget_data = {
            'Canal': ['Télévision', 'Radio', 'Presse'],
            'Budget Total': [
                df['TV_Ad_Budget'].sum(),
                df['Radio_Ad_Budget'].sum(),
                df['Newspaper_Ad_Budget'].sum()
            ]
        }
        budget_df = pd.DataFrame(budget_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_budget = px.pie(
                budget_df,
                values='Budget Total',
                names='Canal',
                title="Répartition du Budget Publicitaire",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_budget, use_container_width=True)
        
        with col2:
            # Efficacité relative
            effectiveness = {
                'Canal': ['Télévision', 'Radio', 'Presse'],
                'Efficacité': [
                    df['Sales'].corr(df['TV_Ad_Budget']),
                    df['Sales'].corr(df['Radio_Ad_Budget']),
                    df['Sales'].corr(df['Newspaper_Ad_Budget'])
                ]
            }
            eff_df = pd.DataFrame(effectiveness)
            
            fig_eff = px.bar(
                eff_df,
                x='Canal',
                y='Efficacité',
                title="Efficacité des Canaux (Corrélation avec Ventes)",
                color='Efficacité',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_eff, use_container_width=True)
    
    # Recommandations
    st.subheader("💡 Recommandations Stratégiques")
    
    if all(col in df.columns for col in ['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget', 'Sales']):
        tv_corr = df['Sales'].corr(df['TV_Ad_Budget'])
        radio_corr = df['Sales'].corr(df['Radio_Ad_Budget'])
        newspaper_corr = df['Sales'].corr(df['Newspaper_Ad_Budget'])
        
        best_channel = max([('TV', tv_corr), ('Radio', radio_corr), ('Newspaper', newspaper_corr)], 
                          key=lambda x: x[1])
        
        st.info(f"""
        **🎯 Canal le plus efficace:** **{best_channel[0]}** (corrélation: {best_channel[1]:.3f})
        
        **📝 Recommandations:**
        - 💰 Allouer plus de budget au canal **{best_channel[0]}**
        - 📊 Surveiller l'efficacité du canal **{min([('TV', tv_corr), ('Radio', radio_corr), ('Newspaper', newspaper_corr)], key=lambda x: x[1])[0]}**
        - 🔄 Tester de nouvelles stratégies pour les canaux moins performants
        """)
    
    # Détection d'anomalies
    st.subheader("⚠️ Détection d'Anomalies")
    
    if st.checkbox("Afficher les valeurs aberrantes"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                st.warning(f"**{col}**: {len(outliers)} valeur(s) aberrantes détectée(s)")
                with st.expander(f"Voir les outliers de {col}"):
                    st.dataframe(outliers[[col]])

if __name__ == "__main__":
    main()