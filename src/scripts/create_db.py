import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333/"))
collection_name = "markets"
vector_dim = 768

client.create_collection(
    collection_name=f"{collection_name}",
    vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
)