import json
import requests
from datetime import datetime, timezone
import ast
from typing import Any, Sequence, List, Dict

def _parse_list_field(value: Any) -> List[Any]:
    """
    Accepts a list OR a string like '["Yes","No"]' and returns a Python list.
    Falls back to ast.literal_eval for slightly non-JSON formats.
    """
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
    pairs = [{"outcome": outcomes[i], "price": prices[i]} for i in range(n)]
    return pairs

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
    end_date_only = ""
    if raw_end:
        try:
            end_date_only = datetime.fromisoformat(raw_end.replace("Z", "+00:00")).date().isoformat()
        except Exception:
            end_date_only = raw_end.split("T")[0].strip()

    current_date_only = datetime.now(timezone.utc).date().isoformat()

    outcomes_raw = m.get("outcomes", [])
    prices_raw = m.get("outcomePrices", [])

    outcomes = _parse_list_field(outcomes_raw)
    prices_list = _to_floats(_parse_list_field(prices_raw))

    outcome_pairs = _pair_outcomes_prices(outcomes, prices_list)

    end = end_date_only
    current = current_date_only
    outcomes = _parse_list_field(m.get("outcomes", []))
    prices = _to_floats(_parse_list_field(m.get("outcomePrices", [])))
    outcome_pairs = _pair_outcomes_prices(outcomes, prices)

    # Heuristics
    today = datetime.now(timezone.utc).date()
    end_date_obj = datetime.fromisoformat(end).date()
    time_to_expiry = max((end_date_obj - today).days, 0)

    spread = abs(prices[0] - prices[1]) if len(prices) >= 2 else 0.0
    extremeness = min(prices) if prices else 0.0
    price_sum = sum(prices)
    volume_24h = float(m.get("volume24hr", 0.0))

    return {
        "conditionId": market_id,
        "question": m.get("question", ""),
        "description": m.get("description", ""),
        "endDate": end,
        "currentDate": current,
        "timeToExpiryDays": time_to_expiry,
        "spread": spread,
        "extremeness": extremeness,
        "priceSum": price_sum,
        "volume24h": volume_24h,
        "outcomePairs": outcome_pairs,
        "news": "pass",
    }