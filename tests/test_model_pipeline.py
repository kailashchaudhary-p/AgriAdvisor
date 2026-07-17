import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model_pipeline import get_model_summary, predict_crop, predict_crop_with_confidence, train_and_save_model


def test_training_and_prediction():
    summary = train_and_save_model()
    assert summary["accuracy"] >= 0.0
    crop = predict_crop([90, 42, 43, 20.8, 82.0, 6.5, 202.9])
    assert isinstance(crop, str)
    assert crop


def test_model_summary_and_confidence_prediction():
    summary = get_model_summary()
    assert summary["accuracy"] >= 0.0
    assert summary["algorithm"]
    assert "classification_report" in summary
    assert "confusion_matrix" in summary

    crop, confidence = predict_crop_with_confidence([90, 42, 43, 20.8, 82.0, 6.5, 202.9])
    assert isinstance(crop, str)
    assert 0.0 <= confidence <= 1.0
