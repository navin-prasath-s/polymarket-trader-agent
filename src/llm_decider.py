import json
from typing import List, Dict, Any
from src.models import ask_direction_with_function


INPUT_FILE = "llm_structured_inputs.json"
OUTPUT_FILE = "llm_direction_results.json"


def run_llm_direction_batch(input_path: str) -> List[Dict[str, Any]]:
    """
    Load structured market+news inputs, ask LLM for direction,
    and store simplified decision output per market.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        markets = json.load(f)

    results = []

    for i, market in enumerate(markets, 1):
        try:
            decision = ask_direction_with_function(market)
            result = {
                "conditionId": market.get("conditionId"),
                "question": market.get("question"),
                "llm_decision": decision
            }
            results.append(result)
            print(f"[{i}/{len(markets)}] ✅ {market.get('conditionId')} → {decision['decision']}")
        except Exception as e:
            print(f"[{i}/{len(markets)}] ❌ Error for market {market.get('conditionId')}: {e}")

    return results


if __name__ == "__main__":
    decisions = run_llm_direction_batch(INPUT_FILE)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)

    print(f"\n✅ Saved {len(decisions)} LLM decisions to {OUTPUT_FILE}")
