import streamlit as st

from model_pipeline import get_farming_tips, predict_crop, train_and_save_model

st.set_page_config(page_title="AgriAdvisor", page_icon="🌱", layout="centered")

st.title("🌾 AgriAdvisor")
st.caption("AI-powered crop recommendation system for better farm planning")

st.markdown("---")

with st.sidebar:
    st.header("⚙️ Setup")
    if st.button("Train model"):
        with st.spinner("Training the recommendation model..."):
            summary = train_and_save_model()
        st.success(f"Model training completed with accuracy: {summary['accuracy']:.4f}")

    st.markdown("---")
    st.write("Enter soil and weather values to get a crop recommendation.")

col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=70)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=145, value=40)
    potassium = st.number_input("Potassium (K)", min_value=0, max_value=205, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=120.0)

if st.button("🌱 Recommend Crop", type="primary"):
    try:
        crop_name = predict_crop([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall])
        st.success(f"✅ Recommended Crop: **{crop_name}**")

        st.info("### 🌿 Farming Guidance")
        for tip in get_farming_tips(crop_name):
            st.write(f"- {tip}")
    except Exception as exc:
        st.error(f"Unable to generate recommendation right now. Error: {exc}")

st.markdown("---")
st.caption("Built with Python, Streamlit, scikit-learn, and joblib")