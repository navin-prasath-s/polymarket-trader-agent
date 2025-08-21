from uuid import uuid5, NAMESPACE_URL
import queue
import threading

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
        self._market_queue = queue.Queue()
        self._worker_thread = None
        self._running = True
        self._start_worker()

    def _start_worker(self):
        """Start the background worker thread for processing markets."""
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        logger.info("Market processing worker thread started")

    def _process_queue(self):
        """Background worker that processes markets from the queue one batch at a time."""
        while self._running:
            try:
                markets = self._market_queue.get(timeout=1)
                logger.info(f"Worker processing {len(markets)} markets from queue")
                self._process_markets(markets)
                self._market_queue.task_done()
                logger.info(f"Worker completed processing {len(markets)} markets")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker thread processing markets: {e}")

    def _process_markets(self, markets):
        """Process a batch of markets (embedding + upserting)."""
        processed_count = 0
        for i, market in enumerate(markets):
            try:
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
                processed_count += 1

                # Progress logging every 10 markets
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(markets)} markets")

            except Exception as e:
                logger.error(f"Error processing market {market.get('condition_id', 'unknown')}: {e}")
                continue

        logger.info(f"Completed processing batch: {processed_count}/{len(markets)} markets successfully processed")

    def on_market_added(self, data: dict) -> None:
        markets = data.get("markets", [])

        # Limit to first N markets if max_markets is specified
        if self.max_markets is not None:
            markets = markets[:self.max_markets]

        queue_size = self._market_queue.qsize()
        print(f"on_market_added called with {len(markets)} markets (queue size: {queue_size})")

        # Add to queue instead of processing immediately
        self._market_queue.put(markets)
        print(f"Markets queued for processing. Queue size now: {self._market_queue.qsize()}")

    def on_market_resolved(self, data: dict) -> None:
        markets = data.get("markets", [])
        print(f"on_market_resolved called with data: {len(markets)}")
        if markets:
            try:
                point_ids = [generate_uuid(market["condition_id"]) for market in markets]
                client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=point_ids),
                    wait=True,
                )
                logger.info(f"Deleted {len(markets)} resolved markets from database")
            except Exception as e:
                logger.error(f"Error deleting resolved markets: {e}")
        print(f"markets_resolved: {len(markets)}")

    def on_payout_logs(self, data: dict) -> None:
        pass

    def shutdown(self):
        """Gracefully shutdown the worker thread."""
        logger.info("Shutting down market processor...")
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Market processor shutdown complete")


handler = QdrantHandler()
wl = WebhookListener(port=8001, path="/market_event")
wl.set_handler(handler)

try:
    wl.start()
    input("Listening (Press Enter to stop)\n")
finally:
    wl.stop()
    handler.shutdown()