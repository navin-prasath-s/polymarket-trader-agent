from uuid import uuid5, NAMESPACE_URL
import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient, models

load_dotenv()
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
collection_name = "markets"

vector_dim = 768


def generate_uuid(string: str) -> str:
    """
    Generate a UUID based on a string using the NAMESPACE_URL namespace.
    """
    return str(uuid5(NAMESPACE_URL, string))

record = {
    "condition_id": "0x11a367fea8169759c13b9dbf4547e6fb8aaecfd22f87674870f5940e86c90bdb",
    "question": "Will it rain tomorrow?",
    "description": "Resolves YES if...",
    "tokens": ["YES", "NO"],
}





# # Insert
# combined_text = f"{record['question']}\n\n{record['description']}"
# dummy_vector = [0.0] * vector_dim
#
# point = {
#     "id": generate_uuid(record["condition_id"]),
#     "vector": dummy_vector,
#     "payload": {
#         "condition_id": record["condition_id"],
#         "question": record["question"],
#         "description": record["description"],
#         "text": combined_text,
#         "tokens": record["tokens"],
#     },
# }
#
# client.upsert(collection_name, points=[point])
# print("Inserted:", point["id"])



# #Fetch
# pts = client.retrieve(
#     collection_name=collection_name,
#     ids=[generate_uuid(record["condition_id"])],
#     with_payload=True,
#     with_vectors=True,
# )
# print("retrieve by ID:", pts)


# # Delete
# point_id = client.delete(
#     collection_name=collection_name,
#     points_selector=models.PointIdsList(points=[generate_uuid(record["condition_id"])]),
#     wait=True,
# )
# print("deleted by ID:", point_id)


#
# scroll_filter = None
# offset = None
#
# while True:
#     result = client.scroll(
#         collection_name=collection_name,
#         limit=100,              # adjust batch size
#         with_vectors=False,     # do NOT return vectors
#         with_payload=True,
#         offset=offset
#     )
#
#     points, offset = result
#
#     for point in points:
#         print(point.payload)  # Just print payload dict
#
#     if offset is None:  # No more data
#         break