
from unittest.mock import Mock, patch

import pytest

from knowledge_assistant.ingestion.fetcher import fetch_document


def test_rejects_unapproved_url():
    with pytest.raises(ValueError, match="approved HTTPS"):
        fetch_document("https://example.com/private")


@patch("knowledge_assistant.ingestion.fetcher.httpx.get")
def test_extracts_document_metadata(mock_get):
    response = Mock()
    response.url = "https://docs.python.org/3/tutorial/index.html"
    response.text = """
    <html>
      <head><title>Python Tutorial</title></head>
      <body>
        <nav>Navigation that should be removed</nav>
        <main><h1>Introduction</h1><p>Python is a programming language.</p></main>
      </body>
    </html>
    """
    response.raise_for_status.return_value = None
    mock_get.return_value = response

    document = fetch_document("https://docs.python.org/3/tutorial/index.html")

    assert document.title == "Python Tutorial"
    assert "Python is a programming language." in document.text
    assert "Navigation that should be removed" not in document.text
    assert document.url == "https://docs.python.org/3/tutorial/index.html"
