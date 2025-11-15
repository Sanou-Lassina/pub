# Prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def main():
    st.markdown("""
    <style>
    .prediction-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="prediction-header">🔮 Prédictions des Ventes</h1>', unsafe_allow_html=True)
    
    # Vérification des prérequis
    if not st.session_state.get('models_trained', False):
        st.warning("""
        ⚠️ **Prérequis manquants:**
        
        Veuillez d'abord:
        1. 📁 Importer vos données dans la page **Accueil**
        2. 🤖 Entraîner les modèles dans la page **Performances**
        3. 🎯 Sélectionner un modèle optimal
        """)
        return
    
    if not st.session_state.get('selected_model'):
        st.warning("⚠️ Veuillez d'abord sélectionner un modèle dans l'onglet 'Recommandations' de la page Performances")
        return
    
    # Onglets pour les prédictions
    tab1, tab2, tab3 = st.tabs([
        "🎯 Prédiction Manuelle", 
        "📊 Prédiction par Fichier", 
        "📈 Analyse des Prédictions"
    ])
    
    with tab1:
        show_manual_prediction()
    
    with tab2:
        show_file_prediction()
    
    with tab3:
        show_prediction_analysis()

def show_manual_prediction():
    st.header("🎯 Prédiction Manuelle")
    
    # Interface de saisie des budgets
    st.subheader("💰 Saisie des Budgets Publicitaires")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tv_budget = st.number_input(
            "Budget TV ($)",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=10.0,
            help="Budget publicitaire alloué à la télévision"
        )
    
    with col2:
        radio_budget = st.number_input(
            "Budget Radio ($)",
            min_value=0.0,
            max_value=500.0,
            value=25.0,
            step=5.0,
            help="Budget publicitaire alloué à la radio"
        )
    
    with col3:
        newspaper_budget = st.number_input(
            "Budget Newspaper ($)",
            min_value=0.0,
            max_value=500.0,
            value=20.0,
            step=5.0,
            help="Budget publicitaire alloué à la presse"
        )
    
    # Bouton de prédiction
    if st.button("🔮 Prédire les Ventes", type="primary"):
        try:
            # Préparation des données d'entrée
            input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
            
            # Normalisation
            scaler = st.session_state.scaler
            input_scaled = scaler.transform(input_data)
            
            # Prédiction
            model = st.session_state.best_model
            prediction = model.predict(input_scaled)[0]
            
            # Affichage du résultat
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style='color: white; text-align: center;'>📊 Résultat de la Prédiction</h3>
                <div style='text-align: center; font-size: 2.5rem; font-weight: bold; margin: 1rem 0;'>
                    ${prediction:.2f}
                </div>
                <p style='text-align: center;'>Ventes prédites pour les budgets saisis</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Analyse de la prédiction
            show_prediction_analysis_details(tv_budget, radio_budget, newspaper_budget, prediction)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
    
    # Prédictions rapides avec des scénarios prédéfinis
    st.subheader("🚀 Scénarios Rapides")
    
    scenarios = {
        "📺 Focus TV": [200, 10, 10],
        "📻 Focus Radio": [50, 100, 10],
        "📰 Focus Presse": [50, 10, 100],
        "⚖️ Mix Équilibré": [100, 50, 50],
        "💰 Budget Serré": [50, 25, 25]
    }
    
    cols = st.columns(len(scenarios))
    
    for i, (scenario_name, budgets) in enumerate(scenarios.items()):
        with cols[i]:
            if st.button(scenario_name, key=f"scenario_{i}"):
                # Mise à jour des valeurs
                st.session_state.tv_budget = budgets[0]
                st.session_state.radio_budget = budgets[1]
                st.session_state.newspaper_budget = budgets[2]
                
                # Déclencher la prédiction
                st.rerun()

def show_prediction_analysis_details(tv_budget, radio_budget, newspaper_budget, prediction):
    """Affiche l'analyse détaillée d'une prédiction"""
    
    st.subheader("📈 Analyse de la Prédiction")
    
    # ROI par canal
    total_budget = tv_budget + radio_budget + newspaper_budget
    roi_total = prediction / total_budget if total_budget > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Budget Total", f"${total_budget:.2f}")
    
    with col2:
        st.metric("Ventes Prédites", f"${prediction:.2f}")
    
    with col3:
        st.metric("ROI Global", f"{roi_total:.2f}x")
    
    with col4:
        st.metric("Marge", f"${prediction - total_budget:.2f}")
    
    # Répartition du budget
    st.subheader("📊 Répartition du Budget")
    
    budget_data = {
        'Canal': ['TV', 'Radio', 'Presse'],
        'Budget': [tv_budget, radio_budget, newspaper_budget],
        'Pourcentage': [
            (tv_budget / total_budget) * 100,
            (radio_budget / total_budget) * 100,
            (newspaper_budget / total_budget) * 100
        ]
    }
    
    budget_df = pd.DataFrame(budget_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            budget_df,
            values='Budget',
            names='Canal',
            title="Répartition du Budget",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(
            budget_df,
            x='Canal',
            y='Pourcentage',
            title="Pourcentage par Canal",
            text='Pourcentage',
            color='Canal'
        )
        fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Recommandations d'optimisation
    st.subheader("💡 Recommandations d'Optimisation")
    
    # Calcul de l'efficacité relative basée sur les corrélations historiques
    if st.session_state.df is not None:
        df = st.session_state.df
        tv_corr = df['Sales'].corr(df['TV_Ad_Budget']) if 'TV_Ad_Budget' in df.columns else 0
        radio_corr = df['Sales'].corr(df['Radio_Ad_Budget']) if 'Radio_Ad_Budget' in df.columns else 0
        newspaper_corr = df['Sales'].corr(df['Newspaper_Ad_Budget']) if 'Newspaper_Ad_Budget' in df.columns else 0
        
        efficacite_actuelle = (
            (tv_budget / total_budget) * tv_corr +
            (radio_budget / total_budget) * radio_corr +
            (newspaper_budget / total_budget) * newspaper_corr
        )
        
        # Scénario optimisé
        budget_optimise = [0, 0, 0]
        if tv_corr > radio_corr and tv_corr > newspaper_corr:
            budget_optimise = [total_budget * 0.7, total_budget * 0.2, total_budget * 0.1]
        elif radio_corr > tv_corr and radio_corr > newspaper_corr:
            budget_optimise = [total_budget * 0.2, total_budget * 0.7, total_budget * 0.1]
        else:
            budget_optimise = [total_budget * 0.2, total_budget * 0.1, total_budget * 0.7]
        
        efficacite_optimisee = (
            (budget_optimise[0] / total_budget) * tv_corr +
            (budget_optimise[1] / total_budget) * radio_corr +
            (budget_optimise[2] / total_budget) * newspaper_corr
        )
        
        if efficacite_optimisee > efficacite_actuelle:
            st.info(f"""
            **🎯 Opportunité d'Optimisation Détectée**
            
            Votre allocation actuelle a une efficacité de **{efficacite_actuelle:.3f}**
            Une allocation optimisée pourrait atteindre **{efficacite_optimisee:.3f}**
            
            **Recommandation:**
            - TV: ${budget_optimise[0]:.2f} ({budget_optimise[0]/total_budget*100:.1f}%)
            - Radio: ${budget_optimise[1]:.2f} ({budget_optimise[1]/total_budget*100:.1f}%)
            - Presse: ${budget_optimise[2]:.2f} ({budget_optimise[2]/total_budget*100:.1f}%)
            """)

def show_file_prediction():
    st.header("📊 Prédiction par Fichier")
    
    st.info("""
    **📝 Format attendu:**
    Votre fichier Excel doit contenir les colonnes:
    - `TV_Ad_Budget` - Budget publicitaire télévision
    - `Radio_Ad_Budget` - Budget publicitaire radio  
    - `Newspaper_Ad_Budget` - Budget publicitaire presse
    """)
    
    # Upload du fichier
    uploaded_file = st.file_uploader(
        "Choisissez un fichier Excel avec les budgets publicitaires",
        type=['xlsx', 'xls'],
        key="prediction_file"
    )
    
    if uploaded_file is not None:
        try:
            # Lecture du fichier
            df_input = pd.read_excel(uploaded_file)
            
            # Vérification des colonnes requises
            required_cols = ['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget']
            missing_cols = [col for col in required_cols if col not in df_input.columns]
            
            if missing_cols:
                st.error(f"❌ Colonnes manquantes: {', '.join(missing_cols)}")
                return
            
            # Aperçu des données
            st.subheader("👀 Aperçu des Données d'Entrée")
            st.dataframe(df_input.head(), use_container_width=True)
            
            # Bouton de prédiction
            if st.button("🔮 Lancer les Prédictions sur le Fichier", type="primary"):
                with st.spinner("Calcul des prédictions..."):
                    # Préparation des données
                    X_input = df_input[required_cols].values
                    
                    # Normalisation
                    scaler = st.session_state.scaler
                    X_input_scaled = scaler.transform(X_input)
                    
                    # Prédictions
                    model = st.session_state.best_model
                    predictions = model.predict(X_input_scaled)
                    
                    # Création du dataframe de résultats
                    df_results = df_input.copy()
                    df_results['Sales_Predicted'] = predictions
                    df_results['Prediction_Date'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Affichage des résultats
                    st.subheader("📋 Résultats des Prédictions")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Statistiques des prédictions
                    st.subheader("📊 Statistiques des Prédictions")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Nombre de Prédictions", len(predictions))
                    
                    with col2:
                        st.metric("Ventes Moyennes Prédites", f"${predictions.mean():.2f}")
                    
                    with col3:
                        st.metric("Ventes Totales Prédites", f"${predictions.sum():.2f}")
                    
                    with col4:
                        st.metric("Budget Total", f"${df_input[required_cols].sum().sum():.2f}")
                    
                    # Téléchargement des résultats
                    st.subheader("💾 Téléchargement des Résultats")
                    
                    # Conversion en Excel
                    excel_buffer = pd.ExcelWriter('predictions_results.xlsx', engine='xlsxwriter')
                    df_results.to_excel(excel_buffer, index=False, sheet_name='Predictions')
                    excel_buffer.close()
                    
                    with open('predictions_results.xlsx', 'rb') as file:
                        st.download_button(
                            label="📥 Télécharger les Résultats (Excel)",
                            data=file,
                            file_name="predictions_ventes_publicitaires.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                    
                    # Conversion en CSV
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger les Résultats (CSV)",
                        data=csv,
                        file_name="predictions_ventes_publicitaires.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier: {str(e)}")

def show_prediction_analysis():
    st.header("📈 Analyse des Prédictions")
    
    if not st.session_state.get('models_trained', False):
        st.warning("⚠️ Aucun modèle entraîné pour l'analyse")
        return
    
    # Récupération des données de test
    df = st.session_state.df
    results = st.session_state.models_results
    selected_model = st.session_state.selected_model
    
    if selected_model not in results:
        st.error("❌ Modèle sélectionné non trouvé")
        return
    
    model_results = results[selected_model]
    y_test_pred = model_results['predictions']['test']
    
    # Préparation des données de test (nécessite de recréer le split)
    from sklearn.model_selection import train_test_split
    
    features = st.session_state.features
    X = df[features]
    y = df['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Graphique des prédictions vs valeurs réelles
    st.subheader("📊 Prédictions vs Valeurs Réelles")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test, y=y_test_pred,
        mode='markers',
        name='Prédictions',
        marker=dict(color='blue', opacity=0.6, size=8)
    ))
    
    # Ligne de perfection
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        name='Ligne parfaite',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=f"Performance du Modèle {selected_model}",
        xaxis_title="Valeurs Réelles ($)",
        yaxis_title="Prédictions ($)",
        showlegend=True,
        width=800,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des résidus
    st.subheader("📈 Analyse des Résidus")
    
    residuals = y_test - y_test_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des résidus
        fig_resid_hist = px.histogram(
            x=residuals,
            nbins=30,
            title="Distribution des Résidus",
            labels={'x': 'Résidus', 'y': 'Fréquence'}
        )
        fig_resid_hist.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_resid_hist, use_container_width=True)
    
    with col2:
        # Résidus vs Prédictions
        fig_resid_scatter = px.scatter(
            x=y_test_pred, y=residuals,
            title="Résidus vs Prédictions",
            labels={'x': 'Prédictions', 'y': 'Résidus'}
        )
        fig_resid_scatter.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_resid_scatter, use_container_width=True)
    
    # Métriques de performance détaillées
    st.subheader("🎯 Métriques de Performance Détaillées")
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    
    with col2:
        st.metric("RMSE", f"${rmse:.2f}")
    
    with col3:
        st.metric("MAE", f"${mae:.2f}")
    
    with col4:
        st.metric("MSE", f"${mse:.2f}")
    
    # Intervalle de confiance
    st.subheader("📊 Intervalle de Confiance des Prédictions")
    
    confidence_level = st.slider("Niveau de confiance (%)", 80, 99, 95)
    
    # Calcul de l'intervalle de confiance basé sur les résidus
    std_residuals = np.std(residuals)
    z_score = {80: 1.28, 90: 1.645, 95: 1.96, 99: 2.576}
    margin_of_error = z_score.get(confidence_level, 1.96) * std_residuals
    
    st.info(f"""
    **Intervalle de Confiance à {confidence_level}%:**
    
    Pour une prédiction de **$X**, l'intervalle de confiance est:
    **${-margin_of_error:.2f}** à **${margin_of_error:.2f}** autour de la prédiction
    
    Cela signifie qu'avec {confidence_level}% de confiance, la valeur réelle se situe dans cet intervalle.
    """)

if __name__ == "__main__":
    main()