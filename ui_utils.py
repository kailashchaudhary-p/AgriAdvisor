import streamlit as st

from model_pipeline import FEATURE_COLUMNS, get_model_summary


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #07111f 0%, #0f172a 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.18), rgba(56, 189, 248, 0.15));
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 20px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1rem 1.15rem;
            min-height: 120px;
        }
        .success-card {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(4, 120, 87, 0.12));
            border: 1px solid rgba(34, 197, 94, 0.35);
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            margin-top: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, icon: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="margin:0 0 0.25rem 0;">{icon} {title}</h4>
            <div style="font-size:1.35rem;font-weight:700;">{value}</div>
            <div style="color:#cbd5e1;font-size:0.9rem;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_info() -> None:
    with st.sidebar:
        st.title("🌾 AgriAdvisor")
        st.caption("AI-powered crop recommendation")
        st.markdown("---")
        summary = get_model_summary()
        st.info(f"Model accuracy: {summary['accuracy']:.4f}")
        st.caption(f"Available crops: {len(summary['crops'])}")
        st.caption(f"Features: {', '.join(FEATURE_COLUMNS)}")
        st.markdown("---")
        st.caption("Navigate the workspace using the pages below.")
        st.page_link("app.py", label="🏠 Home", icon="🏠")
        st.page_link("pages/01_crop_recommendation.py", label="🌱 Crop Recommendation", icon="🌱")
        st.page_link("pages/02_dataset_explorer.py", label="📊 Dataset Explorer", icon="📊")
        st.page_link("pages/03_data_visualization.py", label="📈 Data Visualization", icon="📈")
        st.page_link("pages/04_model_performance.py", label="🧠 Model Performance", icon="🧠")
        st.page_link("pages/05_about.py", label="ℹ️ About", icon="ℹ️")


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <h1 style="margin:0 0 0.25rem 0;">{title}</h1>
            <p style="margin:0;color:#e2e8f0;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
