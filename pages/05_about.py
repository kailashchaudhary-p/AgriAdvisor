import streamlit as st

from ui_utils import apply_theme, render_page_header, render_sidebar_info

apply_theme()
render_sidebar_info()

render_page_header(
    "ℹ️ About AgriAdvisor",
    "A professional crop recommendation system built with Python, Streamlit, scikit-learn, and modern data visualization.",
)

st.markdown("### Project Overview")
st.write(
    "AgriAdvisor combines soil, weather, and environmental inputs to recommend the most suitable crop for a field. "
    "The application offers a complete workflow for training, analyzing, visualizing, and evaluating a machine learning model."
)

st.markdown("---")

st.markdown("### Key Features")
col1, col2 = st.columns(2, gap="large")
with col1:
    st.info("- Multi-page Streamlit experience")
    st.info("- Real crop recommendation workflow")
    st.info("- Dataset inspection and summary tools")
with col2:
    st.info("- Professional charts and visual analysis")
    st.info("- Model performance evaluation")
    st.info("- Clean and responsive UI")

st.markdown("---")

st.markdown("### Technology Stack")
st.write("Python, Streamlit, pandas, scikit-learn, matplotlib, seaborn, and joblib")
