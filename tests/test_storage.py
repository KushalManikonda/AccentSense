import numpy as np
from unittest.mock import patch

from utils.storage import save_audio_and_prediction


@patch("utils.storage.history_collection")
@patch("utils.storage.sf.write")
def test_audio_saved_and_db_updated(
    mock_sf_write,
    mock_history_collection
):

    dummy_audio = np.random.randn(16000).astype(np.float32)

    save_audio_and_prediction(
        audio_np=dummy_audio,
        label="telugu",
        confidence=0.95,
        user_id="testuser"
    )

    mock_sf_write.assert_called_once()

    mock_history_collection.insert_one.assert_called_once()


@patch("utils.storage.history_collection")
@patch("utils.storage.sf.write")
def test_confidence_stored_as_float(
    mock_sf_write,
    mock_history_collection
):

    dummy_audio = np.random.randn(16000).astype(np.float32)

    save_audio_and_prediction(
        audio_np=dummy_audio,
        label="malayalam",
        confidence=0.88,
        user_id="abc123"
    )

    inserted_data = mock_history_collection.insert_one.call_args[0][0]

    assert isinstance(inserted_data["confidence"], float)


@patch("utils.storage.history_collection")
@patch("utils.storage.sf.write")
def test_label_saved_correctly(
    mock_sf_write,
    mock_history_collection
):

    dummy_audio = np.random.randn(16000).astype(np.float32)

    save_audio_and_prediction(
        audio_np=dummy_audio,
        label="kannada",
        confidence=0.91,
        user_id="user1"
    )

    inserted_data = mock_history_collection.insert_one.call_args[0][0]

    assert inserted_data["accent"] == "kannada"