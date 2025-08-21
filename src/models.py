import logging
from pathlib import Path

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
cache_dir = project_root / "models"


embedding_model_name = "BAAI/bge-base-en-v1.5"
embedding_model = TextEmbedding(embedding_model_name,
                                cache_dir=str(cache_dir))