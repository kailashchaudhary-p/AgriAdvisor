import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from model_pipeline import FEATURE_COLUMNS, get_feature_importance_df, load_dataset
from ui_utils import apply_theme, render_page_header, render_sidebar_info

apply_theme()
render_sidebar_info()

render_page_header(
    "📈 Data Visualization",
    "Explore the crop dataset through professional charts and visual summaries.",
)

df = load_dataset()

st.markdown("### Crop Distribution")
fig, ax = plt.subplots(figsize=(8, 4))
counts = df["label"].value_counts().head(12)
counts.plot(kind="bar", color="#22c55e", edgecolor="#065f46", ax=ax)
ax.set_title("Crop Distribution")
ax.set_xlabel("Crop")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("---")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### Feature Importance")
    importance_df = get_feature_importance_df()
    fig, ax = plt.subplots(figsize=(7, 4))
    importance_df.plot(kind="bar", x="Feature", y="Importance", color="#38bdf8", ax=ax)
    ax.set_title("Random Forest Feature Importance")
    ax.set_ylabel("Importance")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    corr = df[FEATURE_COLUMNS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown("### Temperature vs Humidity")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["temperature"], df["humidity"], alpha=0.7, color="#f59e0b")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.set_title("Temperature vs Humidity")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col4:
    st.markdown("### Soil pH Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["ph"], bins=15, color="#8b5cf6", edgecolor="black")
    ax.set_xlabel("Soil pH")
    ax.set_ylabel("Frequency")
    ax.set_title("Soil pH Histogram")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
