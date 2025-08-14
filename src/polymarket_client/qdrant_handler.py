from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient

from src.logger import logger
from src.polymarket_client.webhook_listener import WebhookListener, MarketEventHandler
from src.models import embedding_model


client = QdrantClient(url="http://localhost:6663/")
collection_name = "markets"



def generate_uuid(string: str) -> str:
    """
    Generate a UUID based on a string using the NAMESPACE_URL namespace.
    """
    return str(uuid5(NAMESPACE_URL, string))



class QdrantHandler(MarketEventHandler):
    def on_market_added(self, data: dict) -> None:

        for market in data:
            combined_text = f"{market['question']}\n\n{market['description']}"
            vector = embedding_model.embed([combined_text])[0]
            point = {
                "id": generate_uuid(market["condition_id"]),
                "vector": vector,
                "payload": {
                    "condition_id": market["condition_id"],
                    "question": market["question"],
                    "description": market["description"],
                    "text": combined_text,
                    "tokens": market["tokens"],
                },
            }
            client.upsert(collection_name, points=[point])

        logger.info(f"markets_added: {len(data)}")


    def on_market_resolved(self, data: dict) -> None:
        logger.info(f"markets_resolved: {len(data)}")

    def on_payout_logs(self, data: dict) -> None:
        logger.info(f"payout_logs: {len(data)}")


handler = QdrantHandler()
wl = WebhookListener(port=8001, path="/market-event")
wl.set_handler(handler)

wl.start()
input("Listening\n")
wl.stop()