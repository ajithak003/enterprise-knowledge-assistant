
"""Run the ingestion pipeline against one public documentation page."""

import json

from knowledge_assistant.ingestion.pipeline import IngestionPipeline


def main() -> None:
    url = "https://docs.python.org/3/tutorial/index.html"
    pipeline = IngestionPipeline()

    try:
        result = pipeline.ingest_url(url)
        print(json.dumps(result, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
