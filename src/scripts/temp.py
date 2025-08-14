import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
collection_name = "markets"


scroll_filter = None
offset = None

while True:
    result = client.scroll(
        collection_name=collection_name,
        limit=100,              # adjust batch size
        with_vectors=False,     # do NOT return vectors
        with_payload=True,
        offset=offset
    )

    points, offset = result

    for point in points:
        print(point.payload)  # Just print payload dict

    if offset is None:  # No more data
        break