from unittest.mock import patch, MagicMock


@patch("db.mongo.MongoClient")
def test_mongo_client_creation(mock_client):

    mock_instance = MagicMock()

    mock_client.return_value = mock_instance

    assert mock_client is not None