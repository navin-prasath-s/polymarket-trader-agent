import json
import requests
from datetime import datetime, timezone
import ast
from typing import Any, Sequence, List, Dict

def _parse_list_field(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return ast.literal_eval(s)
    return [value] if value not in (None, "") else []

def _to_floats(seq: Sequence[Any]) -> List[float]:
    out = []
    for x in seq:
        if isinstance(x, (int, float)):
            out.append(float(x))
        elif isinstance(x, str):
            out.append(float(x.strip()))
        else:
            out.append(float(str(x)))
    return out

def _pair_outcomes_prices(outcomes: Sequence[Any], prices: Sequence[float]) -> List[Dict[str, Any]]:
    n = min(len(outcomes), len(prices))
    return [{"outcome": outcomes[i], "price": prices[i]} for i in range(n)]

def fetch_and_extract(market_id: str) -> dict:
    url = "https://gamma-api.polymarket.com/markets"
    params = {"condition_ids": market_id}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()
    markets = data if isinstance(data, list) else data.get("data", [])
    if not markets:
        raise ValueError(f"No market data for ID {market_id}")

    m = markets[0]

    raw_end = m.get("endDateIso") or m.get("endDate") or ""
    try:
        end_date_only = datetime.fromisoformat(raw_end.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        end_date_only = raw_end.split("T")[0].strip()

    current_date_only = datetime.now(timezone.utc).date().isoformat()

    outcomes = _parse_list_field(m.get("outcomes", []))
    prices = _to_floats(_parse_list_field(m.get("outcomePrices", [])))

    outcome_pairs = _pair_outcomes_prices(outcomes, prices)

    return {
        "question": m.get("question", ""),
        "description": m.get("description", ""),
        "endDate": end_date_only,
        "currentDate": current_date_only,
        "outcomePairs": outcome_pairs
    }
