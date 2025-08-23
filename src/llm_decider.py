import json
import os
from datetime import datetime
from uuid import uuid4
from typing import Any, Dict, List

from src.models import ask_direction_with_function
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
client = QdrantClient(url=QDRANT_URL)
COLLECTION = "llm_decisions"
INPUT_FILE = "llm_structured_inputs.json"
OUTPUT_JSON = "llm_direction_results.json"



def run_and_store(input_path: str, output_path: str) -> None:
    # Load market data
    with open(input_path, "r", encoding="utf-8") as f:
        markets: List[Dict[str, Any]] = json.load(f)

    decisions_summary = []
    points_to_insert = []

    for idx, market in enumerate(markets, 1):
        try:
            decision = ask_direction_with_function(market)

            # Prepare JSON summary record
            summary = {
                "conditionId": market.get("conditionId"),
                "question": market.get("question"),
                "llm_decision": decision
            }
            decisions_summary.append(summary)

            # Prepare Qdrant point insertion
            point = PointStruct(
                id=str(uuid4()),
                vector=[0.0],  # dummy vector for now
                payload={
                    "conditionId": market.get("conditionId"),
                    "decision": decision.get("decision"),
                    "option": decision.get("option"),
                    "direction": decision.get("direction"),
                    "insert_time": datetime.now(timezone.utc),
                    "at_10min": None,
                    "at_30min": None,
                    "at_1hr": None,
                    "at_2hr": None,
                    "at_6hr": None
                }
            )
            points_to_insert.append(point)

            print(f"[{idx}/{len(markets)}] Processed {market.get('conditionId')}: {decision.get('decision')}")
        except Exception as e:
            print(f"[{idx}/{len(markets)}] Error for market {market.get('conditionId')}: {e}")

    # Bulk insert into Qdrant
    if points_to_insert:
        client.upsert(collection_name=COLLECTION, points=points_to_insert, wait=True)
        print(f"\n✅ Inserted {len(points_to_insert)} decisions into Qdrant collection '{COLLECTION}'.")

    # Save JSON summary
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(decisions_summary, f_out, indent=2)
    print(f"✅ Saved {len(decisions_summary)} decisions to JSON file '{output_path}'.")


if __name__ == "__main__":
    run_and_store(INPUT_FILE, OUTPUT_JSON)