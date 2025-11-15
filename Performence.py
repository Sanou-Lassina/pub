# Performence.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import time

def main():
    st.markdown("""
    <style>
    .performance-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="performance-header">🤖 Performance des Modèles</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Veuillez d'abord importer des données dans la page d'Accueil")
        return
    
    df = st.session_state.df
    
    # Vérification des colonnes requises
    required_cols = ['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget', 'Sales']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Colonnes manquantes: {', '.join(missing_cols)}")
        st.info("Veuillez vous assurer que votre dataset contient les colonnes: TV_Ad_Budget, Radio_Ad_Budget, Newspaper_Ad_Budget, Sales")
        return
    
    # Onglets pour l'analyse des performances
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Entraînement", 
        "📊 Comparaison", 
        "🏆 Recommandations", 
        "💾 Modèles"
    ])
    
    with tab1:
        show_model_training(df)
    
    with tab2:
        show_model_comparison()
    
    with tab3:
        show_recommendations()
    
    with tab4:
        show_model_management()

def show_model_training(df):
    st.header("🎯 Entraînement des Modèles")
    
    # Configuration de l'entraînement
    st.subheader("⚙️ Configuration de l'Entraînement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20) / 100
        random_state = st.number_input("Random State", value=42)
    
    with col2:
        features = st.multiselect(
            "Features à utiliser:",
            ['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget'],
            default=['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget']
        )
        
        target = 'Sales'
    
    # Sélection des modèles
    st.subheader("🤖 Sélection des Modèles")
    
    models_to_train = {}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.checkbox("Régression Linéaire", value=True):
            models_to_train['Linear Regression'] = LinearRegression()
        
        if st.checkbox("Ridge Regression"):
            alpha_ridge = st.number_input("Alpha Ridge", value=1.0, key="alpha_ridge")
            models_to_train['Ridge Regression'] = Ridge(alpha=alpha_ridge)
    
    with col2:
        if st.checkbox("Lasso Regression"):
            alpha_lasso = st.number_input("Alpha Lasso", value=1.0, key="alpha_lasso")
            models_to_train['Lasso Regression'] = Lasso(alpha=alpha_lasso)
        
        if st.checkbox("Random Forest"):
            n_estimators = st.number_input("N Estimators", value=100, key="n_estimators")
            models_to_train['Random Forest'] = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    
    with col3:
        if st.checkbox("Gradient Boosting"):
            learning_rate = st.number_input("Learning Rate", value=0.1, key="learning_rate")
            models_to_train['Gradient Boosting'] = GradientBoostingRegressor(
                n_estimators=100, learning_rate=learning_rate, random_state=random_state
            )
    
    # Préparation des données
    X = df[features]
    y = df[target]
    
    # Normalisation des features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Bouton d'entraînement
    if st.button("🚀 Lancer l'Entraînement", type="primary"):
        if not models_to_train:
            st.warning("⚠️ Veuillez sélectionner au moins un modèle")
            return
        
        st.session_state.models_trained = True
        st.session_state.models_results = {}
        st.session_state.scaler = scaler
        st.session_state.features = features
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Entraînement des modèles
        for i, (name, model) in enumerate(models_to_train.items()):
            status_text.text(f"Entraînement de {name}...")
            
            # Entraînement du modèle
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Prédictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calcul des métriques
            train_metrics = {
                'R2': r2_score(y_train, y_pred_train),
                'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'MAE': mean_absolute_error(y_train, y_pred_train),
                'MSE': mean_squared_error(y_train, y_pred_train)
            }
            
            test_metrics = {
                'R2': r2_score(y_test, y_pred_test),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'MAE': mean_absolute_error(y_test, y_pred_test),
                'MSE': mean_squared_error(y_test, y_pred_test)
            }
            
            # Stockage des résultats
            st.session_state.models_results[name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'training_time': training_time,
                'predictions': {
                    'train': y_pred_train,
                    'test': y_pred_test
                }
            }
            
            progress_bar.progress((i + 1) / len(models_to_train))
        
        status_text.text("✅ Entraînement terminé!")
        st.balloons()
    
    # Affichage des résultats si l'entraînement est fait
    if st.session_state.get('models_trained', False):
        st.subheader("📈 Résultats de l'Entraînement")
        
        for name, results in st.session_state.models_results.items():
            with st.expander(f"📊 {name}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Train", f"{results['train_metrics']['R2']:.4f}")
                    st.metric("R² Test", f"{results['test_metrics']['R2']:.4f}")
                
                with col2:
                    st.metric("RMSE Train", f"{results['train_metrics']['RMSE']:.2f}")
                    st.metric("RMSE Test", f"{results['test_metrics']['RMSE']:.2f}")
                
                with col3:
                    st.metric("MAE Train", f"{results['train_metrics']['MAE']:.2f}")
                    st.metric("MAE Test", f"{results['test_metrics']['MAE']:.2f}")
                
                with col4:
                    st.metric("Temps d'entraînement", f"{results['training_time']:.2f}s")
                
                # Graphique des prédictions vs réelles
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=y_test, y=results['predictions']['test'],
                    mode='markers',
                    name='Prédictions vs Réelles',
                    marker=dict(color='blue', opacity=0.6)
                ))
                
                # Ligne de perfection
                min_val = min(y_test.min(), results['predictions']['test'].min())
                max_val = max(y_test.max(), results['predictions']['test'].max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    name='Ligne parfaite',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{name} - Prédictions vs Valeurs Réelles",
                    xaxis_title="Valeurs Réelles",
                    yaxis_title="Prédictions",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)

def show_model_comparison():
    st.header("📊 Comparaison des Modèles")
    
    if not st.session_state.get('models_trained', False):
        st.warning("⚠️ Veuillez d'abord entraîner les modèles dans l'onglet 'Entraînement'")
        return
    
    results = st.session_state.models_results
    
    # Tableau de comparaison
    st.subheader("📋 Tableau Comparatif")
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Modèle': name,
            'R² Train': result['train_metrics']['R2'],
            'R² Test': result['test_metrics']['R2'],
            'RMSE Train': result['train_metrics']['RMSE'],
            'RMSE Test': result['test_metrics']['RMSE'],
            'MAE Train': result['train_metrics']['MAE'],
            'MAE Test': result['test_metrics']['MAE'],
            'Temps (s)': result['training_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.round(4), use_container_width=True)
    
    # Graphiques de comparaison
    st.subheader("📈 Visualisation des Performances")
    
    metric = st.selectbox("Métrique à comparer:", ['R² Test', 'RMSE Test', 'MAE Test', 'Temps (s)'])
    
    fig = px.bar(
        comparison_df,
        x='Modèle',
        y=metric,
        title=f"Comparaison des Modèles - {metric}",
        color=metric,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Matrice de corrélation des prédictions
    st.subheader("🔗 Corrélation des Prédictions")
    
    # Collecte des prédictions de test
    predictions_df = pd.DataFrame()
    for name, result in results.items():
        predictions_df[name] = result['predictions']['test']
    
    corr_matrix = predictions_df.corr()
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig_heatmap.update_layout(
        title="Corrélation entre les Prédictions des Modèles",
        width=600,
        height=600
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

def show_recommendations():
    st.header("🏆 Recommandations de Modèle")
    
    if not st.session_state.get('models_trained', False):
        st.warning("⚠️ Veuillez d'abord entraîner les modèles dans l'onglet 'Entraînement'")
        return
    
    results = st.session_state.models_results
    
    # Trouver le meilleur modèle basé sur R²
    best_model_name = max(results.items(), key=lambda x: x[1]['test_metrics']['R2'])[0]
    best_model_r2 = results[best_model_name]['test_metrics']['R2']
    
    # Trouver le modèle le plus rapide
    fastest_model_name = min(results.items(), key=lambda x: x[1]['training_time'])[0]
    fastest_model_time = results[fastest_model_name]['training_time']
    
    # Affichage des recommandations
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"""
        **🎯 Meilleur Modèle (Précision)**
        
        **{best_model_name}**
        
        📊 R² Score: **{best_model_r2:.4f}**
        📈 RMSE: **{results[best_model_name]['test_metrics']['RMSE']:.2f}**
        ⏱️ Temps: **{results[best_model_name]['training_time']:.2f}s**
        
        *Recommandé pour la précision maximale*
        """)
    
    with col2:
        st.info(f"""
        **⚡ Modèle le Plus Rapide**
        
        **{fastest_model_name}**
        
        📊 R² Score: **{results[fastest_model_name]['test_metrics']['R2']:.4f}**
        📈 RMSE: **{results[fastest_model_name]['test_metrics']['RMSE']:.2f}**
        ⏱️ Temps: **{fastest_model_time:.2f}s**
        
        *Recommandé pour les applications temps réel*
        """)
    
    # Analyse de trade-off
    st.subheader("📊 Analyse de Trade-off")
    
    tradeoff_data = []
    for name, result in results.items():
        tradeoff_data.append({
            'Modèle': name,
            'R² Test': result['test_metrics']['R2'],
            'Temps (s)': result['training_time']
        })
    
    tradeoff_df = pd.DataFrame(tradeoff_data)
    
    fig_tradeoff = px.scatter(
        tradeoff_df,
        x='Temps (s)',
        y='R² Test',
        text='Modèle',
        title="Trade-off: Précision vs Temps d'Entraînement",
        size_max=60
    )
    
    fig_tradeoff.update_traces(textposition='top center')
    fig_tradeoff.update_layout(
        xaxis_title="Temps d'Entraînement (secondes)",
        yaxis_title="R² Score (Test)"
    )
    
    st.plotly_chart(fig_tradeoff, use_container_width=True)
    
    # Sélection finale du modèle
    st.subheader("🤔 Quel Modèle Choisir?")
    
    selected_model = st.selectbox(
        "Sélectionnez le modèle à utiliser pour les prédictions:",
        list(results.keys()),
        index=list(results.keys()).index(best_model_name)
    )
    
    if st.button("✅ Confirmer la Sélection"):
        st.session_state.selected_model = selected_model
        st.session_state.best_model = results[selected_model]['model']
        st.success(f"✅ Modèle **{selected_model}** sélectionné pour les prédictions!")
        
        # Sauvegarde du modèle
        try:
            joblib.dump(st.session_state.best_model, 'best_model.pkl')
            joblib.dump(st.session_state.scaler, 'scaler.pkl')
            st.info("💾 Modèle et scaler sauvegardés pour les prédictions")
        except Exception as e:
            st.error(f"❌ Erreur lors de la sauvegarde: {str(e)}")

def show_model_management():
    st.header("💾 Gestion des Modèles")
    
    if not st.session_state.get('models_trained', False):
        st.warning("⚠️ Veuillez d'abord entraîner les modèles")
        return
    
    # Information sur les modèles entraînés
    st.subheader("📋 Modèles Entraînés")
    
    for name, result in st.session_state.models_results.items():
        with st.expander(f"🔧 {name}"):
            st.write(f"**Type:** {type(result['model']).__name__}")
            st.write(f"**Paramètres:** {result['model'].get_params()}")
            
            # Bouton de téléchargement
            try:
                model_bytes = joblib.dumps(result['model'])
                st.download_button(
                    label=f"📥 Télécharger {name}",
                    data=model_bytes,
                    file_name=f"{name.replace(' ', '_').lower()}.pkl",
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.error(f"Erreur lors de la sérialisation: {str(e)}")
    
    # Modèle sélectionné
    if st.session_state.get('selected_model'):
        st.subheader("🎯 Modèle Sélectionné")
        
        selected = st.session_state.selected_model
        st.success(f"**Modèle actuellement sélectionné:** {selected}")
        
        # Métriques détaillées du modèle sélectionné
        results = st.session_state.models_results[selected]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("R² Score (Test)", f"{results['test_metrics']['R2']:.4f}")
            st.metric("RMSE (Test)", f"{results['test_metrics']['RMSE']:.2f}")
        
        with col2:
            st.metric("MAE (Test)", f"{results['test_metrics']['MAE']:.2f}")
            st.metric("Temps d'entraînement", f"{results['training_time']:.2f}s")

if __name__ == "__main__":
    main()