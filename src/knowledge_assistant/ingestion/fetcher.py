
"""Fetch and extract readable text from public documentation pages."""

from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class SourceDocument:
    """A fetched document with source metadata."""

    url: str
    title: str
    text: str


ALLOWED_HOSTS = {
    "docs.python.org",
    "fastapi.tiangolo.com",
}


def fetch_document(url: str) -> SourceDocument:
    """Fetch one approved public documentation page."""

    parsed = urlparse(url)

    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"URL is not an approved HTTPS documentation source: {url}")

    response = httpx.get(
        url,
        timeout=20.0,
        follow_redirects=True,
        headers={"User-Agent": "EnterpriseKnowledgeAssistant/0.1"},
    )
    response.raise_for_status()

    # Prevent redirects from taking us to an unapproved host.
    final_url = urlparse(str(response.url))
    if final_url.scheme != "https" or final_url.hostname not in ALLOWED_HOSTS:
        raise ValueError("Redirected to an unapproved URL.")

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup.select(
        "script, style, nav, footer, header, aside, "
        "noscript, .sidebar, .toctree-wrapper"
    ):
        element.decompose()

    main_content = soup.select_one("main") or soup.select_one("article") or soup.body
    if main_content is None:
        raise ValueError(f"No readable content found at {url}")

    title_element = soup.find("title")
    title = (
        title_element.get_text(" ", strip=True)
        if title_element
        else final_url.path.strip("/") or final_url.hostname or "Untitled"
    )

    text = main_content.get_text(" ", strip=True)
    if not text:
        raise ValueError(f"Extracted page is empty: {url}")

    return SourceDocument(
        url=str(response.url),
        title=title,
        text=text,
    )
