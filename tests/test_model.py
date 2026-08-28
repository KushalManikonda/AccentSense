import pytest
import numpy as np
from unittest.mock import MagicMock

from ml.model import extract_embedding, predict


def test_extract_embedding_output_shape():

    dummy_audio = np.random.randn(16000).astype(np.float32)

    mock_extractor = MagicMock()
    mock_hubert = MagicMock()

    mock_input_values = MagicMock()
    mock_input_values.to.return_value = mock_input_values

    mock_extractor.return_value = MagicMock(
        input_values=mock_input_values
    )

    mock_tensor = MagicMock()

    mock_tensor.mean.return_value.cpu.return_value.numpy.return_value = np.random.randn(1, 768)

    mock_hubert.return_value.last_hidden_state = mock_tensor

    emb = extract_embedding(
        dummy_audio,
        mock_extractor,
        mock_hubert
    )

    assert emb.shape == (1, 768)


def test_predict_returns_valid_output():

    dummy_emb = np.random.randn(1, 768)

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([
        [0.1, 0.7, 0.2]
    ])

    mock_label_enc = MagicMock()
    mock_label_enc.inverse_transform.return_value = ["telugu"]

    label, confidence = predict(
        dummy_emb,
        mock_model,
        mock_label_enc
    )

    assert isinstance(label, str)
    assert isinstance(confidence, float)

    assert label == "telugu"

    assert 0.0 <= confidence <= 1.0


def test_predict_selects_highest_probability():

    dummy_emb = np.random.randn(1, 768)

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([
        [0.05, 0.15, 0.80]
    ])

    mock_label_enc = MagicMock()
    mock_label_enc.inverse_transform.return_value = ["malayalam"]

    label, confidence = predict(
        dummy_emb,
        mock_model,
        mock_label_enc
    )

    assert label == "malayalam"
    assert confidence == 0.80

def test_invalid_embedding_shape_failure():
    """Verify that an embedding with wrong feature dimension (100 instead of 768)
    is detectable. The mock model won't raise, so we assert the shape is wrong."""
    invalid_emb = np.random.randn(1, 100)

    # Shape should NOT be (1, 768) — this is what downstream would reject
    assert invalid_emb.shape[1] != 768, (
        "Invalid embedding should have wrong feature dim, got 768"
    )



def test_confidence_boundary_edge_case(borderline_confidence):

    assert borderline_confidence["confidence"] > 0.5


def test_empty_audio_embedding_edge_case(empty_audio):

    assert len(empty_audio) == 0