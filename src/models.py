import logging
from pathlib import Path

from fastembed import TextEmbedding
import ollama

logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
cache_dir = project_root / "models"


embedding_model_name = "BAAI/bge-base-en-v1.5"
embedding_model = TextEmbedding(embedding_model_name,
                                cache_dir=str(cache_dir))


def ask_gemma(question):
    try:
        response = ollama.chat(
            model='gemma3:270m',
            messages=[{
                'role': 'user',
                'content': question
            }]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"