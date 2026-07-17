import streamlit as st

from model_pipeline import get_model_summary, train_and_save_model
from ui_utils import apply_theme, render_metric_card, render_page_header, render_sidebar_info

st.set_page_config(page_title="AgriAdvisor", page_icon="🌱", layout="wide")
apply_theme()
render_sidebar_info()

render_page_header(
    "🌾 AgriAdvisor",
    "An intelligent crop recommendation assistant with analytics, datasets, and model insights.",
)

summary = get_model_summary()

st.markdown("### Welcome to the Agricultural Intelligence Workspace")
st.write(
    "Use the navigation panel to explore the dataset, review model performance, and generate crop recommendations with confidence."
)

col1, col2, col3 = st.columns(3, gap="small")
with col1:
    render_metric_card("Model Accuracy", f"{summary['accuracy']:.4f}", "🎯", "Validated accuracy")
with col2:
    render_metric_card("Number of Crops", len(summary["crops"]), "🌾", "Crop classes available")
with col3:
    render_metric_card("Number of Features", len(summary["features"]), "🧩", "Input variables for prediction")

st.markdown("---")

st.subheader("Quick Actions")

left_col, right_col = st.columns(2, gap="large")
with left_col:
    st.info("Start with the Crop Recommendation page to receive a crop suggestion and confidence score for your soil and climate inputs.")
with right_col:
    st.info("Visit the Dataset Explorer and Data Visualization pages to inspect the data structure and explore relationships in the crop dataset.")

st.markdown("---")

if st.button("🧠 Train Model", type="primary", use_container_width=True):
    with st.spinner("Training the recommendation model..."):
        summary = train_and_save_model()
    st.success(f"Model training completed with accuracy: {summary['accuracy']:.4f}")

st.caption("Built with Python, Streamlit, scikit-learn, pandas, matplotlib, seaborn, and joblib")