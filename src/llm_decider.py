import json
from collections import defaultdict
from typing import Dict, Any, List

from src.determine_if_market_related import process_with_llm
from src.market_matcher import NewsMarketMatcher
from src.get_polymarket_data import fetch_and_extract


def build_structured_llm_input_data() -> List[Dict[str, Any]]:
    """
    Full pipeline: fetch news, match to markets, LLM filter,
    then prepare cleaned structured objects for LLM input (not plain prompts).

    Returns:
        List of market dicts with article list (for LLM context ingestion)
    """
    matcher = NewsMarketMatcher()
    results = matcher.fetch_news_and_match_markets(
        rss_feeds={
            "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
            "NPR": "https://feeds.npr.org/1001/rss.xml"
        },
        state_path="rss_poller_state.json",
        max_items_per_feed=10,
        top_k_markets=5
    )

    llm_results, related_pairs = process_with_llm(results)

    # Group related articles by market ID
    market_to_articles: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for pair in related_pairs:
        cid = pair["condition_id"]
        article = {
            "title": pair["news"].get("title", ""),
            "summary": pair["news"].get("summary", "")
        }
        market_to_articles[cid].append(article)

    final_data = []
    for cid, articles in market_to_articles.items():
        try:
            market_raw = fetch_and_extract(cid)

            # Return full cleaned market data including engineered features
            market_clean = {
                "question": market_raw.get("question", ""),
                "description": market_raw.get("description", ""),
                "endDate": market_raw.get("endDate", ""),
                "currentDate": market_raw.get("currentDate", ""),
                "conditionId": cid,
                "timeToExpiryDays": market_raw.get("timeToExpiryDays", 0),
                "spread": market_raw.get("spread", 0.0),
                "extremeness": market_raw.get("extremeness", 0.0),
                "priceSum": market_raw.get("priceSum", 0.0),
                "volume24h": market_raw.get("volume24h", 0.0),
                "outcomePairs": market_raw.get("outcomePairs", []),
                "related_articles": articles
            }

            final_data.append(market_clean)
        except Exception:
            continue

    return final_data


if __name__ == "__main__":
    llm_ready_data = build_structured_llm_input_data()

    with open("llm_structured_inputs.json", "w", encoding="utf-8") as f:
        json.dump(llm_ready_data, f, indent=2)

    print(f"✅ Saved {len(llm_ready_data)} structured inputs to llm_structured_inputs.json")
