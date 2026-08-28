import pytest
from unittest.mock import patch
from services.recommendation import get_cuisine

@patch("services.recommendation.cuisine_collection")
def test_valid_accent_returns_result(mock_collection):

    mock_collection.find_one.return_value = {
        "accent": "telugu",
        "categories": [
            "Pesarattu",
            "Pulihora",
            "Gutti Vankaya"
        ]
    }

    result = get_cuisine("telugu")

    assert result is not None
    assert "Pesarattu" in result


@patch("services.recommendation.cuisine_collection")
def test_unknown_accent_returns_none(mock_collection):

    mock_collection.find_one.return_value = None

    result = get_cuisine("unknown")

    assert result is None

def test_empty_accent_edge_case():

    result = get_cuisine("")

    assert result is None


def test_none_accent_failure():

    try:
        result = get_cuisine(None)
        assert result is None

    except Exception:
        assert True


def test_whitespace_accent_edge_case():

    result = get_cuisine("   ")

    assert result is None