import json
import logging
import os
from pathlib import Path
import re
from typing import Dict, Any

from dotenv import load_dotenv
from fastembed import TextEmbedding
import ollama
import openai

from src.get_polymarket_data import fetch_and_extract

logger = logging.getLogger(__name__)
load_dotenv()


project_root = Path(__file__).resolve().parent.parent
cache_dir = project_root / "models"


embedding_model_name = "BAAI/bge-base-en-v1.5"
embedding_model = TextEmbedding(embedding_model_name,
                                cache_dir=str(cache_dir))




def ask_llm_if_related(question):
    prompt_asking_if_related = """You are a strict financial analyst. You will be given a news article and a polymarket question.
    Your task is to determine if the news article will have any DIRECT impact on the market outcome.

    IMPORTANT: Be very strict. Most news will NOT impact most markets.

    Examples of IMPACT (yes):
    - News: "Tesla reports record quarterly deliveries" + Market: "Will Tesla stock exceed $200?" = IMPACT (yes) - same company
    - News: "Apple CEO announces new iPhone features" + Market: "Will Apple beat Q4 earnings?" = IMPACT (yes) - same company  
    - News: "Fed cuts interest rates" + Market: "Will Fed cut rates again?" = IMPACT (yes) - same topic

    Examples of NO IMPACT (no):
    - News: "Tesla factory opens in Germany" + Market: "Will Bitcoin reach $100k?" = NO IMPACT (no) - different topics
    - News: "NASA launches Mars rover" + Market: "Will Bitcoin price exceed $100k?" = NO IMPACT (no) - completely unrelated
    - News: "Heavy rainfall in California" + Market: "Will Trump win the election?" = NO IMPACT (no) - unrelated topics
    - News: "Apple releases new iPhone" + Market: "Will Google stock rise?" = NO IMPACT (no) - different companies

    STRICT RULES:
    - Same company/person/organization = IMPACT (yes)
    - Same specific topic/event = IMPACT (yes) 
    - Different companies = NO IMPACT (no)
    - Different topics = NO IMPACT (no)
    - Vague connections = NO IMPACT (no)

    Default to NO IMPACT unless there is a clear, direct connection.
    Only answer "yes" or "no"."""

    try:
        messages = [
            {
                'role': 'system',
                'content': prompt_asking_if_related
            },
            {
                'role': 'user',
                'content': question + "\n\nDoes this news have DIRECT impact on this specific market? Be strict. Answer yes only if clearly connected, otherwise no:"
            }
        ]
        response = ollama.chat(
            model='gemma3:12b',
            messages=messages
        )

        raw_response = response['message']['content']

        cleaned_response = raw_response.lower().strip()
        cleaned_response = re.sub(r'[^\w\s]', '', cleaned_response).strip()

        if cleaned_response == 'yes':
            return 'yes'
        elif cleaned_response == 'no':
            return 'no'
        else:
            print(f"Cleaned response: '{cleaned_response}'")
            return 'invalid_response'

    except Exception as e:
        return f"Error: {e}"



client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

functions = [
    {
        "name": "report_market_direction",
        "description": (
            "Provide a market decision: either say you are 'undecided', or pick a specific outcome and "
            "indicate whether its price is expected to increase or decrease."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["undecided", "decided"],
                    "description": (
                        "'undecided' means no clear signal from the news. "
                        "'decided' means you are confident in a specific outcome and direction."
                    )
                },
                "option": {
                    "type": "string",
                    "description": "Which outcome is supported by the news (e.g., 'Yes', 'No', 'Team A')."
                },
                "direction": {
                    "type": "string",
                    "enum": ["increase", "decrease"],
                    "description": "Do you expect this option's price to go up or down?"
                }
            },
            "required": ["decision"]
        }
    }
]

def ask_direction_with_function(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a market + related news payload, ask the LLM to choose the likely direction.
    """
    prompt = f"""
You are an automated prediction market analyst. Based on the market metadata and news articles, your task is to determine if the market is likely to move due to new information.

Please follow these rules:
- You may answer 'undecided' if the news is ambiguous, unrelated, or lacks strong signals.
- If you do decide, choose one clear outcome (from the market's outcome list) and a direction ('increase' or 'decrease').
- Consider all provided news articles. Your decision should be justifiable by what's in them — do not invent facts.

Market Question:
{payload['question']}

Market Description:
{payload.get('description', 'N/A')}

Outcomes and Prices:
{', '.join(f"{o['outcome']}: {o['price']}" for o in payload.get('outcomePairs', []))}

End Date: {payload.get('endDate')}
Time to Expiry (days): {payload.get('timeToExpiryDays')}
24h Volume: {payload.get('volume24h')}
Spread: {payload.get('spread')}
Extremeness: {payload.get('extremeness')}

News Articles:
""" + "\n".join(
        f"- {a['title']}\n  {a['summary']}" for a in payload.get("related_articles", [])
    )

    # Call OpenAI with function-calling
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",  # or your preferred model
        messages=[
            {"role": "system", "content": "You are an expert market analyst trained to detect news-driven movement."},
            {"role": "user", "content": prompt}
        ],
        functions=functions,
        function_call={"name": "report_market_direction"},
        temperature=0.2,
    )

    fn_call = response.choices[0].message.function_call
    if not fn_call or not fn_call.arguments:
        raise ValueError("Function call failed or missing arguments")

    parsed = json.loads(fn_call.arguments)

    # Always contains at least `decision`
    return parsed




if __name__ == "__main__":
    market_id = "0x6728bcaed6aa840074d7da69cddb04d0f8176592ce197a48f314f873a0ac163b"
    payload = fetch_and_extract(market_id)
    print(payload)
    result = ask_direction_with_function(payload)
    print(json.dumps(result, indent=2))
    # # Example 1 - Should have impact
    # test1 = """
    # News: Apple reports iPhone sales down 10% in China amid increasing competition from local brands like Huawei and Xiaomi.
    #
    # Market: Will Apple's revenue from China be below $15 billion in Q4 2024?
    # """
    #
    # # Example 2 - Should have no impact
    # test2 = """
    # News: NASA successfully launches new Mars rover mission, expected to land in February 2025.
    #
    # Market: Will Bitcoin price exceed $100,000 by end of 2024?
    # """
    #
    # # Example 3 - Should have impact
    # test3 = """
    # News: Federal Reserve announces 0.25% interest rate cut, citing cooling inflation and stable employment numbers.
    #
    # Market: Will the Fed cut interest rates again before December 2024?
    # """
    #
    # # Example 4 - Edge case: indirect impact
    # test4 = """
    # News: Microsoft announces massive layoffs due to AI automation reducing workforce needs.
    #
    # Market: Will Microsoft stock price exceed $400 by year end?
    # """
    #
    # # Test all examples
    # test_cases = [
    #     ("Apple China impact", test1),
    #     ("NASA/Bitcoin no impact", test2),
    #     ("Fed rates impact", test3),
    #     ("Microsoft layoffs impact", test4)
    # ]
    #
    # for name, test_case in test_cases:
    #     result = ask_llm_if_related(test_case)
    #     print(f"{name}: {result}")