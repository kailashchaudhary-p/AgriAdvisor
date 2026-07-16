import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model_pipeline import predict_crop, train_and_save_model


def test_training_and_prediction():
    summary = train_and_save_model()
    assert summary["accuracy"] >= 0.0
    crop = predict_crop([90, 42, 43, 20.8, 82.0, 6.5, 202.9])
    assert isinstance(crop, str)
    assert crop
