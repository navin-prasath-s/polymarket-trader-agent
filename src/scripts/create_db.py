from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6663/")

collection_name = "markets"
vector_dim = 768

client.create_collection(
    collection_name=f"{collection_name}",
    vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
)