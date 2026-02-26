import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json

def get_axis_label(col_name, all_conc_cols):
    """Genera una etiqueta para el eje con unidades."""
    units = ""
    col_name_lower = col_name.lower()
    if 'young' in col_name_lower:
        units = " (Pa)"
    elif 'gravimetr' in col_name_lower:
        units = " (%WET)"
    elif 'transmitan' in col_name_lower: # transmitancia or transmitance
        units = " (Transmitancia)"
    elif col_name in all_conc_cols:
        units = " (mL)"
    return f"{col_name}{units}"

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
Upload your Excel files to compare properties vs. concentration across multiple screenings.
""")

# --- Paso 1: Carga de Datos Múltiples ---
st.sidebar.header("1. Upload Data")

archivos_conc = st.sidebar.file_uploader("Upload Concentrations Excels", type=["xlsx", "xls"], accept_multiple_files=True)
archivos_prop = st.sidebar.file_uploader("Upload Properties Excels (e.g. Young, Transparency)", type=["xlsx", "xls"], accept_multiple_files=True)

if archivos_conc and archivos_prop:
    if len(archivos_conc) != len(archivos_prop):
        st.sidebar.error("⚠️ Sube la misma cantidad de archivos de Concentración y de Propiedades.")
        st.stop()
        
    try:
        archivos_conc = sorted(archivos_conc, key=lambda x: x.name)
        archivos_prop = sorted(archivos_prop, key=lambda x: x.name)

        df_peek = pd.read_excel(archivos_conc[0])
        df_peek.columns = df_peek.columns.astype(str).str.strip()
        
        # --- Paso 2: Configuración Inicial ---
        st.sidebar.header("2. Variable Configuration")
        id_col = st.sidebar.selectbox("Select ID column (common in all files):", df_peek.columns)

        # Obtener lista de columnas de concentración para las unidades
        cols_conc = [c for c in df_peek.columns if c != id_col]

        # --- Paso 3: Calculadora de Composición ---
        st.sidebar.markdown("---") 
        st.sidebar.subheader("🧮 Group Calculator")
        usar_calculadora = st.sidebar.checkbox("Enable calculation (Hydrophilic/Hydrophobic)")

        # --- Paso 4: Procesamiento de Múltiples Archivos ---
        lista_dfs_unidos = []

        for conc_file, prop_file in zip(archivos_conc, archivos_prop):
            df_c = pd.read_excel(conc_file)
            df_p = pd.read_excel(prop_file)
            
            df_c.columns = df_c.columns.astype(str).str.strip()
            df_p.columns = df_p.columns.astype(str).str.strip()

            if usar_calculadora:
                df_c['Total_Hydrophilic'] = 0.0
                df_c['Total_Hydrophobic'] = 0.0
                
                for col in df_c.columns:
                    if col == id_col: continue 

                    match_filico = next((key for key in hydrophilicity if key in col), None)
                    if match_filico:
                        val = pd.to_numeric(df_c[col], errors='coerce').fillna(0)
                        df_c['Total_Hydrophilic'] += val * hydrophilicity[match_filico]

                    match_fobico = next((key for key in lipophilicity if key in col), None)
                    if match_fobico:
                        val = pd.to_numeric(df_c[col], errors='coerce').fillna(0)
                        df_c['Total_Hydrophobic'] += val * lipophilicity[match_fobico]

                df_c['Ratio_Philic_Phobic'] = np.where(
                    df_c['Total_Hydrophobic'] == 0, 0, df_c['Total_Hydrophilic'] / df_c['Total_Hydrophobic']
                )

            df_temp_merged = pd.merge(df_c, df_p, on=id_col, how='inner')
            nombre_limpio = conc_file.name.replace('.xlsx', '').replace('.xls', '')
            df_temp_merged['Screening_File'] = nombre_limpio
            
            lista_dfs_unidos.append(df_temp_merged)

        df_merged = pd.concat(lista_dfs_unidos, ignore_index=True)
        
        if usar_calculadora:
            st.sidebar.success("✅ Multi-file Calculation Complete!")

        # --- Paso 5: Configuración de Visualización Avanzada ---
        st.sidebar.markdown("---")
        st.sidebar.header("3. Advanced Visualization")
        
        cols_disponibles = list(df_merged.columns)
        cols_disponibles_no_id = [c for c in cols_disponibles if c not in [id_col, 'Screening_File']]
        
        x_col = st.sidebar.selectbox("X Axis (Main Variable):", cols_disponibles_no_id, index=0)
        y_col = st.sidebar.selectbox("Y Axis (Property):", cols_disponibles_no_id, index=min(1, len(cols_disponibles_no_id)-1))
        
        # --- NUEVO LÓGICA: Desviación Estándar Condicional para Módulo de Young ---
        parametro_error_x = None
        parametro_error_y = None

        # Verificamos si la palabra 'young' (en cualquier combinación de mayúsculas/minúsculas) está en el eje X
        if 'young' in x_col.lower():
            error_x_col = st.sidebar.selectbox(
                f"Standard Deviation for X-Axis ({x_col}):", 
                ["None"] + cols_disponibles_no_id
            )
            if error_x_col != "None":
                parametro_error_x = error_x_col

        # Verificamos si la palabra 'young' está en el eje Y
        if 'young' in y_col.lower():
            error_y_col = st.sidebar.selectbox(
                f"Standard Deviation for Y-Axis ({y_col}):", 
                ["None"] + cols_disponibles_no_id
            )
            if error_y_col != "None":
                parametro_error_y = error_y_col
                
        st.sidebar.markdown("#### 🎨 Multi-Component Analysis")
        
        numeric_cols = df_merged.select_dtypes(include=np.number).columns.tolist()
        
        multi_color_cols = st.sidebar.multiselect(
            "Select components for composition (No limit):",
            options=[c for c in numeric_cols if c not in [id_col, x_col, y_col, 'Screening_File']],
            help="Select any number of components to see their breakdown."
        )

        st.write("---")
        
        # --- Lógica de Renderizado de Gráficos ---

        # Generar etiquetas con unidades
        x_label = get_axis_label(x_col, cols_conc)
        y_label = get_axis_label(y_col, cols_conc)

        if not multi_color_cols:
            # Vista Estándar
            st.subheader(f"Chart: {y_col} vs {x_col}")
            fig = px.scatter(
                df_merged, 
                x=x_col, 
                y=y_col, 
                error_x=parametro_error_x, # <-- Aplica solo si se seleccionó para X
                error_y=parametro_error_y, # <-- Aplica solo si se seleccionó para Y
                color='Screening_File' if len(archivos_conc) > 1 else None, 
                hover_data=[id_col, 'Screening_File'],
                text=id_col, 
                title=f"Relationship: {y_col} vs {x_col}",
                labels={
                    x_col: x_label,
                    y_col: y_label
                }
            )
            fig.update_traces(
                textposition='top center', 
                marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey'))
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Vista Doble (Con múltiples componentes y screenings)
            st.subheader(f"Multi-Screening Composition: {y_col} vs {x_col}")
            
            colors = px.colors.qualitative.Plotly * (len(multi_color_cols) // len(px.colors.qualitative.Plotly) + 1)
            color_map = dict(zip(multi_color_cols, colors[:len(multi_color_cols)]))

            df_merged['Dominant_Component'] = df_merged[multi_color_cols].fillna(0).idxmax(axis=1)
            
            hover_text = []
            for index, row in df_merged.iterrows():
                text = f"<b>Sample: {row[id_col]}</b><br>"
                text += f"<b>Screening: {row['Screening_File']}</b><br><br><b>Composition:</b><br>"
                for col in multi_color_cols:
                    text += f"- {col}: {row[col]:.2f}<br>"
                hover_text.append(text)
            df_merged['Composition_Hover'] = hover_text

            fig_scatter = px.scatter(
                df_merged, 
                x=x_col, 
                y=y_col, 
                error_x=parametro_error_x, # <-- Barras de error horizontales (si aplica)
                error_y=parametro_error_y, # <-- Barras de error verticales (si aplica)
                color='Dominant_Component', 
                symbol='Screening_File' if len(archivos_conc) > 1 else None, 
                color_discrete_map=color_map, 
                custom_data=['Composition_Hover'],
                text=id_col, 
                title="Scatter (Color = Component | Shape = Screening)",
                labels={
                    x_col: x_label,
                    y_col: y_label
                }
            )
            fig_scatter.update_traces(
                textposition='top center', 
                marker=dict(size=16, line=dict(width=1, color='DarkSlateGrey')),
                hovertemplate="%{customdata[0]}<br><br><b>X:</b> %{x}<br><b>Y:</b> %{y}<extra></extra>"
            )
            
            df_sorted = df_merged.sort_values(by=['Screening_File', x_col])
            df_sorted['Bar_ID'] = df_sorted['Screening_File'] + " - " + df_sorted[id_col].astype(str)
            
            fig_bars = px.bar(
                df_sorted, 
                x='Bar_ID', 
                y=multi_color_cols,
                color_discrete_map=color_map, 
                title="Composition Breakdown (Grouped by Screening)",
                labels={'value': 'Concentration (mL)', 'variable': 'Component'}
            )
            fig_bars.update_layout(barmode='stack', xaxis_title="Screening - Sample")

            col1, col2 = st.columns((3, 2))
            with col1:
                st.plotly_chart(fig_scatter, use_container_width=True)
            with col2:
                st.plotly_chart(fig_bars, use_container_width=True)

        # --- Tabla de Datos Final ---
        with st.expander("View Unified Multi-Screening Data Table"):
            st.dataframe(df_merged)

    except Exception as e:
        st.error(f"An error occurred while processing: {e}")
        st.write("Hint: Asegúrate de subir los archivos en el orden correcto y que todos compartan la misma estructura de columnas.")

else:
    st.info("Please upload your Excel files in the sidebar to begin. You can select multiple files at once.")