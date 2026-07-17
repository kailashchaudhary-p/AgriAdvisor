import pandas as pd
import streamlit as st

from model_pipeline import get_model_summary
from ui_utils import apply_theme, render_metric_card, render_page_header, render_sidebar_info

apply_theme()
render_sidebar_info()

render_page_header(
    "🧠 Model Performance",
    "Review the trained model's accuracy, evaluation metrics, and classification behavior.",
)

summary = get_model_summary()

col1, col2, col3 = st.columns(3, gap="small")
with col1:
    render_metric_card("Accuracy", f"{summary['accuracy']:.4f}", "🎯", "Overall test accuracy")
with col2:
    render_metric_card("Train Samples", f"{summary['train_test_split']['train_size']}", "🧪", "Rows used for training")
with col3:
    render_metric_card("Test Samples", f"{summary['train_test_split']['test_size']}", "📦", "Rows used for evaluation")

st.markdown("---")

st.subheader("Algorithm")
st.code(summary["algorithm"])

st.markdown("---")

st.subheader("Classification Report")
report_df = pd.DataFrame(summary["classification_report"]).T
st.dataframe(report_df, use_container_width=True)

st.markdown("---")

st.subheader("Confusion Matrix")
cm_df = pd.DataFrame(summary["confusion_matrix"], index=summary["crops"], columns=summary["crops"])
st.dataframe(cm_df, use_container_width=True)

st.markdown("---")

st.subheader("Feature Importance")
importance_df = pd.DataFrame(
    {
        "Feature": summary["features"],
        "Importance": [summary["feature_importances"].get(feature, 0.0) for feature in summary["features"]],
    }
).sort_values("Importance", ascending=False)
st.dataframe(importance_df, use_container_width=True)
