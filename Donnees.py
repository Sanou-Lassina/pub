# Donnees.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def main():
    st.markdown("""
    <style>
    .data-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="data-header">💾 Gestion et Traitement des Données</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Veuillez d'abord importer des données dans la page d'Accueil")
        return
    
    df = st.session_state.df
    
    # Onglets pour la gestion des données
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Données Brutes", 
        "🧹 Nettoyage", 
        "🔧 Transformation", 
        "💾 Export"
    ])
    
    with tab1:
        show_raw_data(df)
    
    with tab2:
        show_data_cleaning(df)
    
    with tab3:
        show_data_transformation(df)
    
    with tab4:
        show_data_export(df)

def show_raw_data(df):
    st.header("📋 Données Brutes")
    
    # Informations générales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dimensions", f"{df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    with col2:
        st.metric("Données Manquantes", f"{df.isnull().sum().sum()}")
    
    with col3:
        st.metric("Valeurs Dupliquées", f"{df.duplicated().sum()}")
    
    # Affichage complet des données avec pagination
    st.subheader("👀 Visualisation des Données")
    
    rows_per_page = st.slider("Lignes par page", 5, 50, 10)
    
    total_pages = (len(df) // rows_per_page) + 1
    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    
    start_idx = (page_number - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
    
    st.caption(f"Affichage des lignes {start_idx + 1} à {min(end_idx, len(df))} sur {len(df)} totales")
    
    # Statistiques par colonne
    st.subheader("📊 Statistiques par Colonne")
    
    selected_col = st.selectbox("Sélectionnez une colonne pour les détails:", df.columns)
    
    if selected_col:
        col_data = df[selected_col]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Type", str(col_data.dtype))
        
        with col2:
            st.metric("Valeurs Uniques", col_data.nunique())
        
        with col3:
            st.metric("Valeurs Manquantes", col_data.isnull().sum())
        
        with col4:
            if pd.api.types.is_numeric_dtype(col_data):
                st.metric("Moyenne", f"{col_data.mean():.2f}")
            else:
                st.metric("Mode", str(col_data.mode().iloc[0] if len(col_data.mode()) > 0 else "N/A"))
        
        # Distribution pour les colonnes numériques
        if pd.api.types.is_numeric_dtype(col_data):
            fig = px.histogram(df, x=selected_col, title=f"Distribution de {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Pour les colonnes catégorielles
            value_counts = col_data.value_counts().head(10)
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                        title=f"Top 10 des valeurs de {selected_col}")
            fig.update_layout(xaxis_title=selected_col, yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

def show_data_cleaning(df):
    st.header("🧹 Nettoyage des Données")
    
    df_clean = df.copy()
    
    # Gestion des valeurs manquantes
    st.subheader("🔍 Gestion des Valeurs Manquantes")
    
    missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
    
    if missing_cols:
        st.warning(f"Colonnes avec valeurs manquantes: {', '.join(missing_cols)}")
        
        for col in missing_cols:
            st.write(f"**{col}**: {df_clean[col].isnull().sum()} valeurs manquantes")
            
            col1, col2 = st.columns(2)
            
            with col1:
                method = st.selectbox(
                    f"Méthode pour {col}",
                    ["Supprimer", "Moyenne", "Médiane", "Mode", "Valeur spécifique"],
                    key=f"method_{col}"
                )
            
            with col2:
                if method == "Valeur spécifique":
                    fill_value = st.number_input(f"Valeur pour {col}", value=0.0, key=f"value_{col}")
                else:
                    fill_value = None
            
            if st.button(f"Appliquer à {col}", key=f"btn_{col}"):
                if method == "Supprimer":
                    df_clean = df_clean.dropna(subset=[col])
                elif method == "Moyenne":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif method == "Médiane":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif method == "Mode":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                elif method == "Valeur spécifique":
                    df_clean[col] = df_clean[col].fillna(fill_value)
                
                st.success(f"Valeurs manquantes de {col} traitées!")
    else:
        st.success("✅ Aucune valeur manquante détectée!")
    
    # Gestion des doublons
    st.subheader("🔍 Gestion des Doublons")
    
    duplicates = df_clean.duplicated().sum()
    st.write(f"Nombre de lignes dupliquées: **{duplicates}**")
    
    if duplicates > 0:
        if st.button("Supprimer les doublons"):
            df_clean = df_clean.drop_duplicates()
            st.success(f"{duplicates} doublons supprimés!")
    
    # Gestion des valeurs aberrantes
    st.subheader("📊 Détection des Valeurs Aberrantes")
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        
        if len(outliers) > 0:
            st.warning(f"**{col}**: {len(outliers)} valeur(s) aberrantes détectée(s)")
            
            if st.checkbox(f"Voir les outliers de {col}", key=f"outliers_{col}"):
                st.dataframe(outliers[[col]])
                
            if st.checkbox(f"Corriger les outliers de {col}", key=f"correct_{col}"):
                method = st.radio(
                    f"Méthode de correction pour {col}",
                    ["Cap aux bornes", "Supprimer", "Remplacer par médiane"],
                    key=f"corr_method_{col}"
                )
                
                if st.button(f"Appliquer la correction à {col}", key=f"apply_corr_{col}"):
                    if method == "Cap aux bornes":
                        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    elif method == "Supprimer":
                        df_clean = df_clean[~((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))]
                    elif method == "Remplacer par médiane":
                        median_val = df_clean[col].median()
                        df_clean[col] = np.where(
                            (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound),
                            median_val,
                            df_clean[col]
                        )
                    
                    st.success(f"Valeurs aberrantes de {col} corrigées!")
    
    # Aperçu des données nettoyées
    st.subheader("📋 Aperçu des Données Nettoyées")
    
    st.dataframe(df_clean.head(), use_container_width=True)
    
    # Sauvegarde dans le session state
    if st.button("💾 Sauvegarder les Données Nettoyées"):
        st.session_state.df = df_clean
        st.session_state.data_cleaned = True
        st.success("Données nettoyées sauvegardées avec succès!")

def show_data_transformation(df):
    st.header("🔧 Transformation des Données")
    
    df_transformed = df.copy()
    
    # Normalisation/Standardisation
    st.subheader("📐 Normalisation des Données")
    
    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Sales' in numeric_cols:
        numeric_cols.remove('Sales')  # On ne normalise pas la variable cible
    
    selected_norm_cols = st.multiselect(
        "Sélectionnez les colonnes à normaliser:",
        numeric_cols,
        default=numeric_cols
    )
    
    norm_method = st.radio(
        "Méthode de normalisation:",
        ["Standardisation (Z-score)", "Normalisation Min-Max", "Normalisation Robust"]
    )
    
    if st.button("Appliquer la Normalisation"):
        for col in selected_norm_cols:
            if norm_method == "Standardisation (Z-score)":
                df_transformed[col] = (df_transformed[col] - df_transformed[col].mean()) / df_transformed[col].std()
            elif norm_method == "Normalisation Min-Max":
                df_transformed[col] = (df_transformed[col] - df_transformed[col].min()) / (df_transformed[col].max() - df_transformed[col].min())
            elif norm_method == "Normalisation Robust":
                Q1 = df_transformed[col].quantile(0.25)
                Q3 = df_transformed[col].quantile(0.75)
                df_transformed[col] = (df_transformed[col] - df_transformed[col].median()) / (Q3 - Q1)
        
        st.success("Normalisation appliquée avec succès!")
    
    # Création de nouvelles features
    st.subheader("🎯 Création de Nouvelles Features")
    
    if all(col in df_transformed.columns for col in ['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget']):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.checkbox("Budget Publicitaire Total"):
                df_transformed['Total_Ad_Budget'] = (
                    df_transformed['TV_Ad_Budget'] + 
                    df_transformed['Radio_Ad_Budget'] + 
                    df_transformed['Newspaper_Ad_Budget']
                )
        
        with col2:
            if st.checkbox("ROI par Canal"):
                df_transformed['ROI_TV'] = df_transformed['Sales'] / df_transformed['TV_Ad_Budget']
                df_transformed['ROI_Radio'] = df_transformed['Sales'] / df_transformed['Radio_Ad_Budget']
                df_transformed['ROI_Newspaper'] = df_transformed['Sales'] / df_transformed['Newspaper_Ad_Budget']
        
        with col3:
            if st.checkbox("Mix Publicitaire (%)"):
                total_budget = df_transformed['TV_Ad_Budget'] + df_transformed['Radio_Ad_Budget'] + df_transformed['Newspaper_Ad_Budget']
                df_transformed['TV_Percentage'] = (df_transformed['TV_Ad_Budget'] / total_budget) * 100
                df_transformed['Radio_Percentage'] = (df_transformed['Radio_Ad_Budget'] / total_budget) * 100
                df_transformed['Newspaper_Percentage'] = (df_transformed['Newspaper_Ad_Budget'] / total_budget) * 100
    
    # Aperçu des données transformées
    st.subheader("📋 Aperçu des Données Transformées")
    st.dataframe(df_transformed.head(), use_container_width=True)
    
    # Sauvegarde
    if st.button("💾 Sauvegarder les Données Transformées"):
        st.session_state.df = df_transformed
        st.success("Données transformées sauvegardées!")

def show_data_export(df):
    st.header("💾 Export des Données")
    
    # Options d'export
    st.subheader("📤 Format d'Export")
    
    export_format = st.selectbox(
        "Choisissez le format d'export:",
        ["CSV", "Excel", "JSON"]
    )
    
    # Personnalisation de l'export
    st.subheader("⚙️ Options d'Export")
    
    include_index = st.checkbox("Inclure l'index", value=False)
    
    # Aperçu avant export
    st.subheader("👀 Aperçu des Données à Exporter")
    st.dataframe(df.head(), use_container_width=True)
    
    # Bouton d'export
    if st.button("🚀 Exporter les Données"):
        try:
            if export_format == "CSV":
                csv = df.to_csv(index=include_index)
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv,
                    file_name="donnees_ventes_publicitaires.csv",
                    mime="text/csv"
                )
            
            elif export_format == "Excel":
                excel_buffer = pd.ExcelWriter("donnees_ventes_publicitaires.xlsx", engine='xlsxwriter')
                df.to_excel(excel_buffer, index=include_index, sheet_name='Donnees')
                excel_buffer.close()
                
                with open("donnees_ventes_publicitaires.xlsx", "rb") as file:
                    st.download_button(
                        label="📥 Télécharger Excel",
                        data=file,
                        file_name="donnees_ventes_publicitaires.xlsx",
                        mime="application/vnd.ms-excel"
                    )
            
            elif export_format == "JSON":
                json_str = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Télécharger JSON",
                    data=json_str,
                    file_name="donnees_ventes_publicitaires.json",
                    mime="application/json"
                )
            
            st.success("✅ Données prêtes pour le téléchargement!")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'export: {str(e)}")
    
    # Résumé des données
    st.subheader("📊 Résumé des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Informations Générales:**
        - 📏 Dimensions: {df.shape[0]} × {df.shape[1]}
        - 🔢 Colonnes numériques: {len(df.select_dtypes(include=[np.number]).columns)}
        - 📝 Colonnes texte: {len(df.select_dtypes(include=['object']).columns)}
        """)
    
    with col2:
        st.info(f"""
        **Qualité des Données:**
        - ✅ Valeurs manquantes: {df.isnull().sum().sum()}
        - 🔄 Doublons: {df.duplicated().sum()}
        - 🎯 Variable cible: {'Sales' if 'Sales' in df.columns else 'Non définie'}
        """)

if __name__ == "__main__":
    main()