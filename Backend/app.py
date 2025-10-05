import streamlit as st
import pandas as pd
import requests
import json
import sys
from component import (
    show_experiment_overview,
    show_volcano_plot,
    show_pca_plot,
)


# ---------------------------------------------------------
# 🌍 STEP 1: Load Experiment Metadata Dynamically
# ---------------------------------------------------------
def get_experiment_info():
    """Get experiment info from query params or use default"""
    # Check if experiment info is passed via query params
    query_params = st.query_params
    
    if 'experiment_data' in query_params:
        try:
            # Decode the experiment data from URL parameter
            experiment_data = json.loads(query_params['experiment_data'])
            return experiment_data
        except Exception as e:
            st.error(f"Error parsing experiment data: {e}")
    
    # Default experiment info (fallback)
    return {
        "osd_id": "OSD-3",
        "glds_id": "GLDS-3",
        "matched_paper": {
            "title": "Innate immune responses of Drosophila melanogaster are altered by spaceflight.",
            "link": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7000411/",
        },
        "metadata": {
            "organism": "Drosophila melanogaster",
            "tissue": "Whole Organism",
            "assay_type": "DNA microarray",
            "mission": "STS-121",
            "description": (
                "Gene expression levels were determined in 3rd instar and adult Drosophila melanogaster "
                "reared during spaceflight to elucidate the molecular mechanisms underpinning immune "
                "response alterations under microgravity."
            ),
            "categories": ["Immune System & Inflammation", "Genomics & Epigenetics"],
        },
        "visualizations": {
            "files_url": "https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/OSD-3/files/"
        },
    }

# Get experiment info dynamically
experiment_info = get_experiment_info()


# ---------------------------------------------------------
# 🧠 STEP 2: Helper to load CSVs automatically
# ---------------------------------------------------------
@st.cache_data
def auto_load_files(files_url: str, osd_id: str):
    """
    Automatically fetch NASA OSDR dataset file list,
    download the main CSVs, and return them as DataFrames.
    """
    try:
        response = requests.get(files_url)
        response.raise_for_status()
        data = response.json()

        if osd_id not in data:
            st.error(f"Unexpected JSON structure from NASA OSDR API. Expected {osd_id} but not found.")
            return {}

        file_entries = data[osd_id]["files"]

        # Helper to find specific files
        def find_file(keyword):
            for name, meta in file_entries.items():
                if keyword.lower() in name.lower() and name.endswith(".csv"):
                    return meta["URL"]
            return None

        urls = {
            "differential": find_file("differential_expression"),
            "pca": find_file("PCA_table"),
            "normalized": find_file("normalized_expression"),
            "sample": find_file("SampleTable"),
        }

        dfs = {}
        for key, url in urls.items():
            if url:
                try:
                    df = pd.read_csv(url)
                    dfs[key] = df
                except Exception as e:
                    st.warning(f"Could not load {key} data: {e}")
            else:
                st.warning(f"No file found for {key}")

        return dfs
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return {}


# ---------------------------------------------------------
# 🚀 STEP 3: Streamlit App Setup
# ---------------------------------------------------------
st.set_page_config(page_title="NASA OSDR Gene Expression Explorer", layout="wide")

# Sidebar
st.sidebar.title("NASA OSDR Dataset Controls")
st.sidebar.markdown(f"**Dataset:** {experiment_info['glds_id']}")
st.sidebar.markdown(f"[🔗 View Dataset Files]({experiment_info['visualizations']['files_url']})")


# ---------------------------------------------------------
# 🧩 STEP 4: Load Data Automatically
# ---------------------------------------------------------
files_url = experiment_info["visualizations"]["files_url"]
osd_id = experiment_info["osd_id"]
dataframes = auto_load_files(files_url, osd_id)

diff_df = dataframes.get("differential")
pca_df = dataframes.get("pca")
norm_df = dataframes.get("normalized")
sample_df = dataframes.get("sample")


# ---------------------------------------------------------
# 🧬 STEP 5: Render All Components
# ---------------------------------------------------------
show_experiment_overview(experiment_info["metadata"], experiment_info["matched_paper"])

# Tabs for better organization
tabs = st.tabs([
    "📊 Volcano Plot",
    "🧭 PCA Plot",
])

# Volcano plot
with tabs[0]:
    show_volcano_plot(diff_df)

# PCA plot
with tabs[1]:
    show_pca_plot(pca_df)
