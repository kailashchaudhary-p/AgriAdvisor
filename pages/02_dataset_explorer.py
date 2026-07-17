import pandas as pd
import streamlit as st

from model_pipeline import load_dataset
from ui_utils import apply_theme, render_metric_card, render_page_header, render_sidebar_info

apply_theme()
render_sidebar_info()

render_page_header(
    "📊 Dataset Explorer",
    "Inspect the crop dataset to understand structure, quality, and distribution.",
)

df = load_dataset()

st.subheader("Dataset Preview")
st.dataframe(df.head(15), use_container_width=True)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4, gap="small")
with col1:
    render_metric_card("Rows", f"{df.shape[0]}", "📏", "Number of records")
with col2:
    render_metric_card("Columns", f"{df.shape[1]}", "🧩", "Number of fields")
with col3:
    render_metric_card("Missing Values", f"{int(df.isnull().sum().sum())}", "⚠️", "Blank cells in the dataset")
with col4:
    render_metric_card("Duplicates", f"{int(df.duplicated().sum())}", "🔁", "Duplicate rows")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Types", "Missing Values", "Duplicates", "Summary Statistics", "Shape"])

with tab1:
    dtypes_df = df.dtypes.rename_axis("Feature").reset_index(name="Data Type")
    st.dataframe(dtypes_df, use_container_width=True)

with tab2:
    missing_df = pd.DataFrame({"Missing Values": df.isnull().sum()}).reset_index().rename(columns={"index": "Feature"})
    st.dataframe(missing_df, use_container_width=True)

with tab3:
    if df.duplicated().sum() > 0:
        st.dataframe(df[df.duplicated(keep=False)].head(20), use_container_width=True)
    else:
        st.success("No duplicate records were found in the dataset.")

with tab4:
    st.dataframe(df.describe(include="all").T, use_container_width=True)

with tab5:
    shape_df = pd.DataFrame({"Rows": [df.shape[0]], "Columns": [df.shape[1]]})
    st.dataframe(shape_df, use_container_width=True)
