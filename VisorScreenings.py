import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json

# Cargar los diccionarios de propiedades
try:
    hydrophilicity = json.load(open('hydrophilicity.json', 'r'))
    lipophilicity = json.load(open('lipophilicity.json', 'r'))
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos .json (hydrophilicity.json, lipophilicity.json).")
    st.stop()

# --- Configuración de la Página ---
st.set_page_config(page_title="Experiment Viewer", layout="wide")
st.title("🔬 Experimental Properties Viewer")
st.markdown("""
This tool unifies your scattered data. 
Upload your Excel files to compare properties vs. concentration.
""")

# --- Paso 1: Carga de Datos ---
st.sidebar.header("1. Upload Data")

archivo_conc = st.sidebar.file_uploader("Upload Concentrations Excel", type=["xlsx", "xls"])
archivo_prop = st.sidebar.file_uploader("Upload Properties Excel (e.g. Young, Transparency)", type=["xlsx", "xls"])

if archivo_conc and archivo_prop:
    # --- Paso 2: Procesamiento ---
    try:
        df_conc = pd.read_excel(archivo_conc)
        df_prop = pd.read_excel(archivo_prop)
        
        # Limpieza de nombres de columnas
        df_conc.columns = df_conc.columns.astype(str).str.strip()
        df_prop.columns = df_prop.columns.astype(str).str.strip()

        # --- Paso 3: Configuración Inicial ---
        st.sidebar.header("2. Variable Configuration")
        id_col = st.sidebar.selectbox("Select ID column (common):", df_conc.columns)

        # --- Paso 4: Calculadora de Composición ---
        st.sidebar.markdown("---") 
        st.sidebar.subheader("🧮 Group Calculator")
        usar_calculadora = st.sidebar.checkbox("Enable calculation (Hydrophilic/Hydrophobic)")
        
        cols_calculadas = []

        if usar_calculadora:
            df_conc['Total_Hydrophilic'] = 0.0
            df_conc['Total_Hydrophobic'] = 0.0
            
            detected_philic = []
            detected_phobic = []

            for col in df_conc.columns:
                if col == id_col: continue 

                # Hidrofilicidad
                match_filico = None
                for key in hydrophilicity:
                    if key in col:
                        match_filico = key
                        break
                
                if match_filico:
                    weight = hydrophilicity[match_filico]
                    val = pd.to_numeric(df_conc[col], errors='coerce').fillna(0)
                    df_conc['Total_Hydrophilic'] += val * weight
                    detected_philic.append(f"{col} (como {match_filico})")

                # Hidrofobicidad
                match_fobico = None
                for key in lipophilicity:
                    if key in col:
                        match_fobico = key
                        break
                
                if match_fobico:
                    weight = lipophilicity[match_fobico]
                    val = pd.to_numeric(df_conc[col], errors='coerce').fillna(0)
                    df_conc['Total_Hydrophobic'] += val * weight
                    detected_phobic.append(f"{col} (como {match_fobico})")

            # Cálculo del Ratio
            df_conc['Ratio_Philic_Phobic'] = np.where(
                df_conc['Total_Hydrophobic'] == 0, 
                0, 
                df_conc['Total_Hydrophilic'] / df_conc['Total_Hydrophobic']
            )

            cols_calculadas = ['Total_Hydrophilic', 'Total_Hydrophobic', 'Ratio_Philic_Phobic']
            st.sidebar.success("✅ Calculation Complete!")
            
        # --- Paso 5: Fusión de Tablas ---
        df_merged = pd.merge(df_conc, df_prop, on=id_col, how='inner')

        # --- Paso 6: Configuración de Visualización Avanzada ---
        st.sidebar.markdown("---")
        st.sidebar.header("3. Advanced Visualization")
        
        cols_disponibles = list(df_merged.columns)
        cols_disponibles_no_id = [c for c in cols_disponibles if c != id_col]
        
        x_col = st.sidebar.selectbox("X Axis (Main Variable):", cols_disponibles_no_id, index=0)
        y_col = st.sidebar.selectbox("Y Axis (Property):", cols_disponibles_no_id, index=min(1, len(cols_disponibles_no_id)-1))
        
        # --- NUEVO: Selector de barras de error ---
        error_y_col = st.sidebar.selectbox(
            "Y-Axis Error Bars (e.g. Standard Deviation):", 
            ["None"] + cols_disponibles_no_id
        )
        
        st.sidebar.markdown("#### 🎨 Multi-Component Analysis")
        
        numeric_cols = df_merged.select_dtypes(include=np.number).columns.tolist()
        multi_color_cols = st.sidebar.multiselect(
            "Select up to 4 components for composition:",
            options=[c for c in numeric_cols if c not in [id_col, x_col, y_col]],
            max_selections=4,
            help="Select components to see their breakdown in the hover tooltip and the adjacent bar chart."
        )

        st.write("---")
        
        # --- Lógica de Renderizado de Gráficos ---
        
        # Determinar si aplicamos barras de error
        parametro_error_y = error_y_col if error_y_col != "None" else None

        if not multi_color_cols:
            # Vista Estándar (Sin multiselect)
            st.subheader(f"Chart: {y_col} vs {x_col}")
            fig = px.scatter(
                df_merged, 
                x=x_col, 
                y=y_col, 
                error_y=parametro_error_y, # <-- AÑADIDO: Barras de error
                hover_data=[id_col],
                text=id_col, 
                title=f"Relationship: {y_col} vs {x_col}",
                trendline="ols"
            )
            fig.update_traces(
                textposition='top center', 
                marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey'))
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Vista Doble (Con multiselect activo)
            st.subheader(f"Multi-Component Analysis: {y_col} vs {x_col}")
            
            # Crear un mapa de colores unificado
            colors = px.colors.qualitative.Plotly * (len(multi_color_cols) // len(px.colors.qualitative.Plotly) + 1)
            color_map = dict(zip(multi_color_cols, colors[:len(multi_color_cols)]))

            # 1. Preparar datos para el Hover y el Color Dominante
            df_merged['Dominant_Component'] = df_merged[multi_color_cols].fillna(0).idxmax(axis=1)
            
            hover_text = []
            for index, row in df_merged.iterrows():
                text = f"<b>Sample: {row[id_col]}</b><br><br><b>Composition:</b><br>"
                for col in multi_color_cols:
                    text += f"- {col}: {row[col]:.2f}<br>"
                hover_text.append(text)
            df_merged['Composition_Hover'] = hover_text

            # 2. Gráfico de Dispersión (Izquierda)
            fig_scatter = px.scatter(
                df_merged, 
                x=x_col, 
                y=y_col, 
                error_y=parametro_error_y, # <-- AÑADIDO: Barras de error
                color='Dominant_Component', 
                color_discrete_map=color_map, 
                custom_data=['Composition_Hover'],
                text=id_col, 
                title="Scatter (Colored by Dominant Component)"
            )
            fig_scatter.update_traces(
                textposition='top center', 
                marker=dict(size=16, line=dict(width=1, color='DarkSlateGrey')),
                hovertemplate="%{customdata[0]}<br><br><b>X:</b> %{x}<br><b>Y:</b> %{y}<extra></extra>"
            )
            
            # 3. Gráfico de Barras Apiladas (Derecha)
            df_sorted = df_merged.sort_values(by=x_col)
            fig_bars = px.bar(
                df_sorted, 
                x=id_col, 
                y=multi_color_cols,
                color_discrete_map=color_map, 
                title="Composition Breakdown (Sorted by X-Axis)",
                labels={'value': 'Concentration', 'variable': 'Component'}
            )
            fig_bars.update_layout(barmode='stack')

            # 4. Despliegue en Columnas
            col1, col2 = st.columns((3, 2))
            with col1:
                st.plotly_chart(fig_scatter, use_container_width=True)
            with col2:
                st.plotly_chart(fig_bars, use_container_width=True)

        # --- Tabla de Datos Final ---
        with st.expander("View Unified Data Table"):
            st.dataframe(df_merged)

    except Exception as e:
        st.error(f"An error occurred while processing: {e}")
        st.write("Hint: Check if your column names in Excel match the keys in the JSON files exactly.")

else:
    st.info("Please upload both Excel files in the sidebar to begin.")