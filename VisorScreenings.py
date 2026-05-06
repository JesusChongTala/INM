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
Upload your Excel files to compare properties vs. concentration across multiple screenings.
""")


# --- Helpers: Molar Ratio Calculator ---

def parse_chemicals_csv(chem_file):
    """Parse Chemicals_calc.csv into a lookup dict {name: {g_per_mL, g_per_mol}}."""
    df = pd.read_csv(chem_file)
    df['g_per_mol'] = df[['g_per_mol_LO', 'g_per_mol_HI']].mean(axis=1)
    chem_dict = {}
    for _, row in df.iterrows():
        name = str(row['name'])
        rho = row['g_per_mL'] if pd.notna(row.get('g_per_mL')) else None
        mw  = row['g_per_mol'] if pd.notna(row['g_per_mol']) else None
        if rho is not None and mw is not None:
            chem_dict[name] = {'g_per_mL': float(rho), 'g_per_mol': float(mw)}
    return chem_dict


def calculate_molar_ratios(df, chem_dict, pdms_vol_col, pdms_type_col=None, pdms_type_fixed=None):
    """
    Add *_ratio columns for every *_mL column relative to the PDMS anchor.

    r_i = (V_i · ρ_i / M_i) / (V_PDMS · ρ_PDMS / M_PDMS)

    PDMS identity is resolved from pdms_type_col (per-row) or pdms_type_fixed (constant).
    Components with a matching *_type column are resolved per-row as well.
    Returns (modified_df, list_of_ratio_column_names).
    """
    df = df.copy()
    V_pdms = pd.to_numeric(df[pdms_vol_col], errors='coerce')

    if pdms_type_col and pdms_type_col in df.columns:
        rho_pdms = df[pdms_type_col].map(lambda t: chem_dict.get(str(t), {}).get('g_per_mL', np.nan))
        M_pdms   = df[pdms_type_col].map(lambda t: chem_dict.get(str(t), {}).get('g_per_mol', np.nan))
        rho_pdms = pd.to_numeric(rho_pdms, errors='coerce')
        M_pdms   = pd.to_numeric(M_pdms, errors='coerce')
    elif pdms_type_fixed and pdms_type_fixed in chem_dict:
        rho_pdms = chem_dict[pdms_type_fixed]['g_per_mL']
        M_pdms   = chem_dict[pdms_type_fixed]['g_per_mol']
    else:
        return df, []

    n_pdms = V_pdms * rho_pdms / M_pdms

    ratio_cols = []
    for col in list(df.columns):
        if col == pdms_vol_col or not col.endswith('_mL'):
            continue

        chem_name = col[:-3]  # strip '_mL'
        type_col  = chem_name + '_type'

        if type_col in df.columns:
            rho_i = pd.to_numeric(
                df[type_col].map(lambda t: chem_dict.get(str(t), {}).get('g_per_mL', np.nan)),
                errors='coerce')
            M_i   = pd.to_numeric(
                df[type_col].map(lambda t: chem_dict.get(str(t), {}).get('g_per_mol', np.nan)),
                errors='coerce')
        elif chem_name in chem_dict:
            rho_i = chem_dict[chem_name]['g_per_mL']
            M_i   = chem_dict[chem_name]['g_per_mol']
        else:
            continue

        V_i = pd.to_numeric(df[col], errors='coerce')
        n_i = V_i * rho_i / M_i

        ratio_col = f'{chem_name}_ratio'
        df[ratio_col] = np.where(n_pdms > 0, np.round(n_i / n_pdms, 4), np.nan)
        ratio_cols.append(ratio_col)

    return df, ratio_cols


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

        # --- Paso 3: Calculadora de Composición ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧮 Group Calculator")
        usar_calculadora = st.sidebar.checkbox("Enable calculation (Hydrophilic/Hydrophobic)")

        # --- Paso 4: Calculadora de Ratios Molares ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚗️ Molar Ratio Calculator")
        usar_ratios = st.sidebar.checkbox("Calculate molar ratios (relative to PDMS)")

        pdms_vol_col    = None
        pdms_type_col   = None
        pdms_type_fixed = None
        chem_dict       = {}
        ratio_cols_global = []

        if usar_ratios:
            chem_upload = st.sidebar.file_uploader("Upload Chemicals_calc.csv", type=["csv"])
            if chem_upload:
                chem_dict = parse_chemicals_csv(chem_upload)

                ml_cols = [c for c in df_peek.columns if c.endswith('_mL')]
                type_cols = [c for c in df_peek.columns if c.endswith('_type')]

                if not ml_cols:
                    st.sidebar.warning("No *_mL columns found in the first concentration file.")
                else:
                    # PDMS volume column
                    pdms_default_idx = next(
                        (i for i, c in enumerate(ml_cols) if 'PDMS' in c or 'pdms' in c.lower()),
                        0)
                    pdms_vol_col = st.sidebar.selectbox(
                        "PDMS anchor column (volume):", ml_cols, index=pdms_default_idx)

                    # PDMS type resolution
                    pdms_type_options = ["(fixed — select below)"] + type_cols
                    pdms_type_sel = st.sidebar.selectbox(
                        "PDMS type column (leave fixed if all rows share one type):",
                        pdms_type_options)

                    if pdms_type_sel == "(fixed — select below)":
                        pdms_type_fixed = st.sidebar.selectbox(
                            "PDMS type (chemical name):",
                            sorted(chem_dict.keys()),
                            index=next(
                                (i for i, k in enumerate(sorted(chem_dict.keys()))
                                 if k.startswith('DMS-')),
                                0))
                    else:
                        pdms_type_col = pdms_type_sel
            else:
                st.sidebar.info("Upload Chemicals_calc.csv to enable ratio calculation.")

        # --- Paso 5: Procesamiento de Múltiples Archivos ---
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
                    df_c['Total_Hydrophobic'] == 0, 0,
                    df_c['Total_Hydrophilic'] / df_c['Total_Hydrophobic']
                )

            if usar_ratios and chem_dict and pdms_vol_col and pdms_vol_col in df_c.columns:
                df_c, ratio_cols = calculate_molar_ratios(
                    df_c, chem_dict, pdms_vol_col,
                    pdms_type_col=pdms_type_col,
                    pdms_type_fixed=pdms_type_fixed)
                if not ratio_cols_global:
                    ratio_cols_global = ratio_cols

            df_temp_merged = pd.merge(df_c, df_p, on=id_col, how='inner')
            nombre_limpio = conc_file.name.replace('.xlsx', '').replace('.xls', '')
            df_temp_merged['Screening_File'] = nombre_limpio

            lista_dfs_unidos.append(df_temp_merged)

        df_merged = pd.concat(lista_dfs_unidos, ignore_index=True)

        if usar_calculadora:
            st.sidebar.success("✅ Multi-file Calculation Complete!")
        if usar_ratios and ratio_cols_global:
            st.sidebar.success(f"✅ {len(ratio_cols_global)} molar ratio column(s) added.")

        # --- Paso 6: Configuración de Visualización Avanzada ---
        st.sidebar.markdown("---")
        st.sidebar.header("3. Advanced Visualization")

        cols_disponibles = list(df_merged.columns)
        cols_disponibles_no_id = [c for c in cols_disponibles if c not in [id_col, 'Screening_File']]

        x_col = st.sidebar.selectbox("X Axis (Main Variable):", cols_disponibles_no_id, index=0)
        y_col = st.sidebar.selectbox("Y Axis (Property):", cols_disponibles_no_id, index=min(1, len(cols_disponibles_no_id)-1))

        parametro_error_x = None
        parametro_error_y = None

        if 'young' in x_col.lower():
            error_x_col = st.sidebar.selectbox(
                f"Standard Deviation for X-Axis ({x_col}):",
                ["None"] + cols_disponibles_no_id
            )
            if error_x_col != "None":
                parametro_error_x = error_x_col

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

        # --- Tabla de Ratios Molares ---
        if usar_ratios and ratio_cols_global:
            ratio_cols_present = [c for c in ratio_cols_global if c in df_merged.columns]
            if ratio_cols_present:
                with st.expander("⚗️ Molar Ratios (relative to PDMS)", expanded=False):
                    st.markdown(
                        "Each value is the molar ratio of the component relative to PDMS:  "
                        r"$r_i = \dfrac{V_i \cdot \rho_i / M_i}{V_\text{PDMS} \cdot \rho_\text{PDMS} / M_\text{PDMS}}$"
                    )
                    display_ratio_cols = [id_col, 'Screening_File'] + ratio_cols_present
                    display_ratio_cols = [c for c in display_ratio_cols if c in df_merged.columns]
                    st.dataframe(
                        df_merged[display_ratio_cols].style.format(
                            {c: "{:.3f}" for c in ratio_cols_present}, na_rep="—"
                        ),
                        use_container_width=True
                    )

                    # Heatmap-style bar chart of mean ratios per screening
                    ratio_means = (
                        df_merged.groupby('Screening_File')[ratio_cols_present]
                        .mean()
                        .reset_index()
                        .melt(id_vars='Screening_File', var_name='Component', value_name='Mean Ratio')
                    )
                    ratio_means['Component'] = ratio_means['Component'].str.replace('_ratio', '', regex=False)
                    fig_ratios = px.bar(
                        ratio_means,
                        x='Component', y='Mean Ratio',
                        color='Screening_File',
                        barmode='group',
                        title='Mean Molar Ratios per Screening (relative to PDMS)',
                        labels={'Mean Ratio': 'Molar ratio (mol/mol PDMS)'}
                    )
                    st.plotly_chart(fig_ratios, use_container_width=True)

        # --- Lógica de Renderizado de Gráficos ---

        if not multi_color_cols:
            st.subheader(f"Chart: {y_col} vs {x_col}")
            fig = px.scatter(
                df_merged,
                x=x_col,
                y=y_col,
                error_x=parametro_error_x,
                error_y=parametro_error_y,
                color='Screening_File' if len(archivos_conc) > 1 else None,
                hover_data=[id_col, 'Screening_File'],
                text=id_col,
                title=f"Relationship: {y_col} vs {x_col}"
            )
            fig.update_traces(
                textposition='top center',
                marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey'))
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
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
                if ratio_cols_global:
                    ratio_cols_present = [c for c in ratio_cols_global if c in df_merged.columns]
                    if ratio_cols_present:
                        text += "<br><b>Molar ratios:</b><br>"
                        for rc in ratio_cols_present:
                            val = row[rc]
                            label = rc.replace('_ratio', '')
                            text += f"- {label}: {val:.3f}<br>" if pd.notna(val) else f"- {label}: —<br>"
                hover_text.append(text)
            df_merged['Composition_Hover'] = hover_text

            fig_scatter = px.scatter(
                df_merged,
                x=x_col,
                y=y_col,
                error_x=parametro_error_x,
                error_y=parametro_error_y,
                color='Dominant_Component',
                symbol='Screening_File' if len(archivos_conc) > 1 else None,
                color_discrete_map=color_map,
                custom_data=['Composition_Hover'],
                text=id_col,
                title="Scatter (Color = Component | Shape = Screening)"
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
                labels={'value': 'Concentration', 'variable': 'Component'}
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
