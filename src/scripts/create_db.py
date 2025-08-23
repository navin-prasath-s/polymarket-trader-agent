import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))

collection_name = "markets"
vector_dim = 768

client.create_collection(
    collection_name=f"{collection_name}",
    vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
)
print("OK")


collection_2 = "llm_decisions"

client.create_collection(
    collection_name=f"{collection_2}",
    vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE)
)

for field in ["conditionId", "option", "direction", "insert_time"]:
    client.create_payload_index(
        collection_name=collection_2,
        field_name=field,
        field_schema="keyword"
    )

print(f"Collection '{collection_2}' is ready.")


