import json
import logging
import os
from pathlib import Path
import re

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
            "Provide market decision: 'undecided' or chosen outcome with direction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["undecided", "decided"]
                },
                "option": {
                    "type": "string",
                    "description": "Chosen outcome, e.g., Yes, Team A"
                },
                "direction": {
                    "type": "string",
                    "enum": ["increase", "decrease"],
                    "description": "Expected price movement"
                }
            },
            "required": ["decision"]
        }
    }
]

def ask_direction_with_function(payload: dict) -> dict:
    """
    Sends data to GPT-5 Nano using function calling.
    Returns the structured response.
    """
    user_prompt = (
    "You are an automated Polymarket trading assistant.\n"
    "\n"
    "Behavioral rules:\n"
    "1) Always respond by calling the function 'report_market_direction'. "
    "   Never output free text.\n"
    "2) You may choose 'undecided' if signals are weak/ambiguous or you do not "
    "   expect a significant probability shift. Do not force a decision.\n"
    "3) If you do decide, choose exactly one existing outcome label from the provided data, "
    "   and a single direction: 'increase' or 'decrease'.\n"
    "4) Consider only the provided market snapshot and prompt. Do NOT invent external facts.\n"
    "5) Prefer caution near market close, on thin liquidity, or when price already implies the view.\n"
    "6) Ignore style; optimize for correctness and calibration. No chain-of-thought in the output.\n"
    "\n"
    "Data schema reminder (input):\n"
    "{\n"
    '  \"question\": str,\n'
    '  \"description\": str,\n'
    '  \"endDate\": \"YYYY-MM-DD\",\n'
    '  \"currentDate\": \"YYYY-MM-DD\",\n'
    '  \"outcomePairs\": [ {\"outcome\": str, \"price\": float}, ... ]\n'
    "}\n"
    "\n"
    "Decision policy (high-level heuristics; not strict rules):\n"
    "- Price context: extremely low/high prices may be near-saturated; require stronger evidence to predict further move.\n"
    "- Time context: if endDate is very near and no strong catalyst is implied in the prompt, lean 'undecided'.\n"
    "- Multi-outcome parity: if outcomes are close and no differentiator is present, lean 'undecided'.\n"
    "- Binary example: If 'Yes' looks underpriced relative to prompt signals, recommend option='Yes', direction='increase'.\n"
)
    messages = [
        {"role": "system", "content": "Respond only by calling the function."},
        {"role": "user", "content": json.dumps({
            "data": payload,
            "prompt": user_prompt
        })}
    ]

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        functions=functions,
        function_call={"name": "report_market_direction"},
    )

    msg = response.choices[0].message
    if msg.function_call:
        return json.loads(msg.function_call.arguments)
    return {"decision": "undecided"}




if __name__ == "__main__":
    market_id = "0x6728bcaed6aa840074d7da69cddb04d0f8176592ce197a48f314f873a0ac163b"
    payload = fetch_and_extract(market_id)
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