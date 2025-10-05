# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import numpy as np

# # ---------------------------------------------------------
# # 1️⃣ Overview + Experiment Info Section
# # ---------------------------------------------------------
# def show_experiment_overview(metadata, paper_info):
#     """Display the high-level study overview (metadata + paper)."""
#     st.title(f"🧬 {paper_info.get('title', 'Experiment Overview')}")
#     st.markdown(f"**Paper:** [{paper_info.get('title', '')}]({paper_info.get('link', '')})")
#     st.markdown("---")

#     st.subheader("📘 Experiment Metadata")
#     cols = st.columns(2)
#     with cols[0]:
#         st.markdown(f"**Organism:** {metadata.get('organism', 'N/A')}")
#         st.markdown(f"**Tissue:** {metadata.get('tissue', 'N/A')}")
#         st.markdown(f"**Assay Type:** {metadata.get('assay_type', 'N/A')}")
#         st.markdown(f"**Mission:** {metadata.get('mission', 'N/A')}")
#     with cols[1]:
#         categories = metadata.get('categories', [])
#         if categories:
#             st.markdown("**Categories:** " + ", ".join(categories))
#         desc = metadata.get("description", "")
#         st.markdown(f"**Description:** {desc[:300]}...")

#     with st.expander("🧠 Scientific Background"):
#         st.markdown("""
#         This study investigates how **spaceflight (microgravity)** affects the **innate immune system**
#         in *Drosophila melanogaster* (fruit fly).  
#         Using a **DNA microarray assay**, gene expression profiles were analyzed for flies reared
#         aboard the **STS-121 Space Shuttle mission**, comparing **spaceflight vs ground control** groups.

#         Results from the paper revealed that:
#         - Immune response genes (e.g., antimicrobial peptides) were **downregulated**.
#         - **Oxidative stress** and **cytoskeletal remodeling** genes were **upregulated**.
#         - The Drosophila immune model is a **valid proxy** for understanding human immune changes in microgravity.
#         """)


# # ---------------------------------------------------------
# # 2️⃣ Volcano Plot Component
# # ---------------------------------------------------------
# def show_volcano_plot(diff_expr_df):
#     """Render a volcano plot with explanations."""
#     if diff_expr_df is None or diff_expr_df.empty:
#         st.warning("No differential expression data available.")
#         return

#     st.subheader("🌋 Differential Gene Expression (Volcano Plot)")
#     diff_expr_df.columns = diff_expr_df.columns.str.strip().str.lower()

#     logfc_col = next((c for c in diff_expr_df.columns if "log" in c and "fc" in c), None)
#     pval_col = next((c for c in diff_expr_df.columns if "p" in c and "val" in c), None)
#     gene_col = next((c for c in diff_expr_df.columns if "gene" in c or "symbol" in c), None)

#     if not logfc_col or not pval_col:
#         st.error("Could not detect 'logFC' or 'p-value' columns in data.")
#         return

#     logfc_threshold = st.sidebar.slider("Log2 Fold Change threshold", 0.0, 5.0, 1.0, 0.1)
#     pval_threshold = st.sidebar.number_input("Max adjusted p-value", 0.0, 1.0, 0.05)

#     df = diff_expr_df.dropna(subset=[logfc_col, pval_col])
#     filtered = df[(df[logfc_col].abs() >= logfc_threshold) & (df[pval_col] <= pval_threshold)]

#     st.write(f"**Filtered Genes:** {len(filtered)}")

#     fig = px.scatter(
#         filtered,
#         x=logfc_col,
#         y=-np.log10(filtered[pval_col]),
#         hover_data=[gene_col] if gene_col else None,
#         color=filtered[logfc_col].apply(lambda x: "Upregulated" if x > 0 else "Downregulated"),
#         color_discrete_map={"Upregulated": "red", "Downregulated": "blue"},
#         title="Volcano Plot of Differentially Expressed Genes",
#         labels={logfc_col: "Log2 Fold Change", pval_col: "-log10(Adjusted p-value)"}
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     with st.expander("🧩 Interpretation"):
#         st.markdown("""
#         - Each point = one gene  
#         - **X-axis:** log2 fold change (how much expression changed in space)  
#         - **Y-axis:** significance (−log10 adjusted p-value)  
#         - **Red (right):** upregulated in spaceflight  
#         - **Blue (left):** downregulated  
        
#         Genes appearing far from center are the **most affected by spaceflight**.
#         """)


# # ---------------------------------------------------------
# # 3️⃣ PCA Visualization Component
# # ---------------------------------------------------------
# def show_pca_plot(pca_df):
#     """Render a 3D PCA plot with explanation."""
#     if pca_df is None or pca_df.empty:
#         st.warning("No PCA data available.")
#         return

#     st.subheader("🧭 Principal Component Analysis (PCA)")
#     pca_df.columns = pca_df.columns.str.strip().str.lower()

#     if "sample" not in pca_df.columns:
#         pca_df["sample"] = [f"Sample_{i}" for i in range(1, len(pca_df) + 1)]

#     pcs = [c for c in pca_df.columns if c.startswith("pc")]
#     if len(pcs) < 3:
#         st.warning("PCA file lacks sufficient PC columns (pc1, pc2, pc3).")
#         return

#     fig = px.scatter_3d(
#         pca_df,
#         x=pcs[0],
#         y=pcs[1],
#         z=pcs[2],
#         hover_name="sample",
#         title="3D PCA Clustering of Samples",
#         color_discrete_sequence=["#1f77b4"]
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     with st.expander("🧩 Interpretation"):
#         st.markdown("""
#         PCA reduces complex gene expression data to three main dimensions (PC1, PC2, PC3).  
#         - **Close points** = samples with similar expression profiles  
#         - **Separated clusters** = distinct biological states (e.g., space vs ground)  
        
#         Clear separation implies **strong gene expression differences** caused by microgravity.
#         """)


# # ---------------------------------------------------------
# # 4️⃣ Heatmap (Optional Extension)
# # ---------------------------------------------------------
# def show_heatmap(normalized_df, top_n=30):
#     """Render a heatmap of top differential genes."""
#     import seaborn as sns
#     import matplotlib.pyplot as plt

#     if normalized_df is None or normalized_df.empty:
#         st.warning("No normalized expression data available.")
#         return

#     st.subheader("🔥 Top Differential Genes (Heatmap)")
#     normalized_df.columns = normalized_df.columns.str.strip().str.lower()

#     # Pick first few genes for visual clarity
#     subset = normalized_df.head(top_n)
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.heatmap(subset.set_index(subset.columns[0]), cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

#     with st.expander("🧩 Interpretation"):
#         st.markdown("""
#         This heatmap shows expression patterns for the most variable genes.  
#         - **Red:** upregulated  
#         - **Blue:** downregulated  
#         Clusters highlight groups of genes responding similarly to spaceflight conditions.
#         """)


import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------
# 1️⃣ Overview + Experiment Info Section
# ---------------------------------------------------------
def show_experiment_overview(metadata, paper_info):
    """Display the high-level study overview (metadata + paper)."""
    st.title(f"🧬 {paper_info.get('title', 'Experiment Overview')}")
    st.markdown(f"**Paper:** [{paper_info.get('title', '')}]({paper_info.get('link', '')})")
    st.markdown("---")

    st.subheader("📘 Experiment Metadata")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"**Organism:** {metadata.get('organism', 'N/A')}")
        st.markdown(f"**Tissue:** {metadata.get('tissue', 'N/A')}")
        st.markdown(f"**Assay Type:** {metadata.get('assay_type', 'N/A')}")
        st.markdown(f"**Mission:** {metadata.get('mission', 'N/A')}")
    with cols[1]:
        categories = metadata.get('categories', [])
        if categories:
            st.markdown("**Categories:** " + ", ".join(categories))
        desc = metadata.get("description", "")
        st.markdown(f"**Description:** {desc[:400]}...")

    with st.expander("🧠 Scientific Background"):
        st.markdown("""
        This study explores how **microgravity during spaceflight** alters gene expression in
        *Drosophila melanogaster* (fruit fly). Using **DNA microarrays**, researchers compared
        flies reared aboard the **STS-121 Space Shuttle** to those on Earth.
        
        Findings:
        - **Downregulation** of immune-related pathways (Toll, Imd)
        - **Upregulation** of oxidative-stress genes
        - Cytoskeletal and adhesion genes also affected

        These changes mirror known **immune suppression** observed in astronauts, validating
        Drosophila as a **model organism** for studying immune adaptation to spaceflight.
        """)


# ---------------------------------------------------------
# 2️⃣ Volcano Plot Component
# ---------------------------------------------------------
def show_volcano_plot(diff_expr_df):
    """Render a volcano plot with explanations."""
    if diff_expr_df is None or diff_expr_df.empty:
        st.warning("No differential expression data available.")
        return

    st.subheader("🌋 Differential Gene Expression (Volcano Plot)")
    diff_expr_df.columns = diff_expr_df.columns.str.strip().str.lower()

    logfc_col = next((c for c in diff_expr_df.columns if "log" in c and "fc" in c), None)
    pval_col = next((c for c in diff_expr_df.columns if "p" in c and "val" in c), None)
    gene_col = next((c for c in diff_expr_df.columns if "gene" in c or "symbol" in c), None)

    if not logfc_col or not pval_col:
        st.error("Could not detect 'logFC' or 'p-value' columns in data.")
        return

    logfc_threshold = st.sidebar.slider("Log2 Fold Change threshold", 0.0, 5.0, 1.0, 0.1)
    pval_threshold = st.sidebar.number_input("Max adjusted p-value", 0.0, 1.0, 0.05)

    df = diff_expr_df.dropna(subset=[logfc_col, pval_col])
    filtered = df[(df[logfc_col].abs() >= logfc_threshold) & (df[pval_col] <= pval_threshold)]

    st.write(f"**Filtered Genes:** {len(filtered)}")

    fig = px.scatter(
        filtered,
        x=logfc_col,
        y=-np.log10(filtered[pval_col]),
        hover_data=[gene_col] if gene_col else None,
        color=filtered[logfc_col].apply(lambda x: "Upregulated" if x > 0 else "Downregulated"),
        color_discrete_map={"Upregulated": "red", "Downregulated": "blue"},
        title="Volcano Plot of Differentially Expressed Genes",
        labels={logfc_col: "Log2 Fold Change", pval_col: "-log10(Adjusted p-value)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🧩 Interpretation"):
        st.markdown("""
        - Each point = one gene  
        - **X-axis:** log2 fold change (change in space vs ground)  
        - **Y-axis:** −log10(adjusted p-value)  
        - **Red:** upregulated in spaceflight  
        - **Blue:** downregulated  
        Genes far from center are the most affected by microgravity.
        """)


# ---------------------------------------------------------
# 3️⃣ PCA Visualization Component
# ---------------------------------------------------------
def show_pca_plot(pca_df):
    """Render a 3D PCA plot with explanation."""
    if pca_df is None or pca_df.empty:
        st.warning("No PCA data available.")
        return

    st.subheader("🧭 Principal Component Analysis (PCA)")
    pca_df.columns = pca_df.columns.str.strip().str.lower()

    if "sample" not in pca_df.columns:
        pca_df["sample"] = [f"Sample_{i}" for i in range(1, len(pca_df) + 1)]

    pcs = [c for c in pca_df.columns if c.startswith("pc")]
    if len(pcs) < 3:
        st.warning("PCA file lacks sufficient PC columns (pc1, pc2, pc3).")
        return

    fig = px.scatter_3d(
        pca_df,
        x=pcs[0],
        y=pcs[1],
        z=pcs[2],
        hover_name="sample",
        title="3D PCA Clustering of Samples",
        color_discrete_sequence=["#1f77b4"]
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🧩 Interpretation"):
        st.markdown("""
        PCA reduces high-dimensional gene expression data into three main components.  
        - **Close points** = similar expression profiles  
        - **Separated clusters** = distinct biological states (e.g., space vs ground)  
        Clear separation means strong transcriptional differences due to microgravity.
        """)

