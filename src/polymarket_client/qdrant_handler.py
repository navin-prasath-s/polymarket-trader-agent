from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv

from src.logger import setup_logging
from src.polymarket_client.webhook_listener import WebhookListener, MarketEventHandler
from src.models import embedding_model


load_dotenv()
logger = setup_logging()

client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
collection_name = "markets"



def generate_uuid(string: str) -> str:
    """
    Generate a UUID based on a string using the NAMESPACE_URL namespace.
    """
    return str(uuid5(NAMESPACE_URL, string))


class QdrantHandler(MarketEventHandler):
    def __init__(self, max_markets: int = None):
        self.max_markets = max_markets

    def on_market_added(self, data: dict) -> None:
        markets = data.get("markets", [])

        # Limit to first N markets if max_markets is specified
        if self.max_markets is not None:
            markets = markets[:self.max_markets]

        print(f"on_market_added called with data: {len(markets)}")

        for market in markets:
            embeddings = list(embedding_model.embed([market['question']]))
            if not embeddings:
                logger.error(f"No embedding generated for market {market['condition_id']}")
                continue
            vector = embeddings[0]
            point = {
                "id": generate_uuid(market["condition_id"]),
                "vector": vector,
                "payload": {
                    "condition_id": market["condition_id"],
                    "question": market["question"],
                    "description": market["description"],
                    "tokens": market["tokens"],
                },
            }
            client.upsert(collection_name, points=[point])
        print(f"markets_added: {len(markets)}")

    def on_market_resolved(self, data: dict) -> None:
        markets = data.get("markets", [])
        print(f"on_market_resolved called with data: {len(markets)}")
        if markets:
            point_ids = [generate_uuid(market["condition_id"]) for market in markets]
            client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=point_ids),
                wait=True,
            )
        print(f"markets_resolved: {len(markets)}")

    def on_payout_logs(self, data: dict) -> None:
        pass


handler = QdrantHandler(max_markets=50)
wl = WebhookListener(port=8001, path="/market_event")
wl.set_handler(handler)

wl.start()
input("Listening\n")
wl.stop()