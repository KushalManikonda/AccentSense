import numpy as np
import soundfile as sf
import io
import pytest

from services.audio_service import (
    read_audio_bytes_to_np,
    np_audio_to_wav_bytes,
    TARGET_SR
)

def test_read_audio_returns_numpy_array():
    sr = 22050
    audio = np.random.randn(sr * 2).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")

    wav_bytes = buffer.getvalue()

    output = read_audio_bytes_to_np(wav_bytes)

    assert isinstance(output, np.ndarray)


def test_audio_resampled_to_target_sr():
    sr = 22050
    audio = np.random.randn(sr * 2).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")

    wav_bytes = buffer.getvalue()

    output = read_audio_bytes_to_np(wav_bytes)

    expected_length = int(len(audio) * TARGET_SR / sr)

    assert abs(len(output) - expected_length) < 5


def test_stereo_audio_converted_to_mono():
    sr = 16000

    stereo_audio = np.random.randn(sr * 2, 2).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, stereo_audio, sr, format="WAV")

    wav_bytes = buffer.getvalue()

    output = read_audio_bytes_to_np(wav_bytes)

    assert output.ndim == 1


def test_np_audio_to_wav_bytes_returns_bytes():
    audio = np.random.randn(16000).astype(np.float32)

    wav_bytes = np_audio_to_wav_bytes(audio)

    assert isinstance(wav_bytes, bytes)


def test_invalid_audio_raises_exception():
    with pytest.raises(Exception):
        read_audio_bytes_to_np(b"invalid_audio")

def test_empty_audio_edge_case(empty_audio):

    assert len(empty_audio) == 0

def test_silent_audio_edge_case(silent_audio):

    assert np.all(silent_audio == 0)


def test_long_audio_edge_case(long_audio):

    assert len(long_audio) == 16000 * 60


def test_invalid_audio_bytes_failure(invalid_audio_bytes):

    with pytest.raises(Exception):
        read_audio_bytes_to_np(invalid_audio_bytes)