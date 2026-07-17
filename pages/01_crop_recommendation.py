import streamlit as st

from model_pipeline import FEATURE_COLUMNS, get_farming_tips, get_model_summary, predict_crop_with_confidence
from ui_utils import apply_theme, render_metric_card, render_page_header, render_sidebar_info

apply_theme()
render_sidebar_info()

render_page_header(
    "🌱 Crop Recommendation",
    "Enter soil and weather conditions to receive a crop suggestion with a confidence score.",
)

summary = get_model_summary()

st.markdown("### Current Model Status")
col1, col2 = st.columns([1, 1], gap="large")
with col1:
    render_metric_card("Model Accuracy", f"{summary['accuracy']:.4f}", "🎯", "Validated on the test split")
with col2:
    render_metric_card("Available Crops", len(summary["crops"]), "🌾", "Crop classes in the trained model")

st.markdown("---")

st.markdown("### Input Parameters")
left_col, right_col = st.columns(2, gap="large")
with left_col:
    nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=70)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=145, value=40)
    potassium = st.number_input("Potassium (K)", min_value=0, max_value=205, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)

with right_col:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=120.0)

if st.button("🌱 Recommend Crop", type="primary", use_container_width=True):
    try:
        crop_name, confidence = predict_crop_with_confidence(
            [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        )
        st.markdown(
            f"""
            <div class="success-card">
                <h3 style="margin:0 0 0.25rem 0;">✅ Recommended Crop</h3>
                <h2 style="margin:0;">{crop_name}</h2>
                <p style="margin:0.4rem 0 0 0;">Confidence score: <strong>{confidence:.1%}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("🌿 Farming Guidance")
        for tip in get_farming_tips(crop_name):
            st.write(f"- {tip}")
    except Exception as exc:
        st.error(f"Unable to generate a recommendation right now. Error: {exc}")

st.caption(f"Input features used: {', '.join(FEATURE_COLUMNS)}")
