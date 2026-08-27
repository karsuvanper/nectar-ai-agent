from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
try:
    collections = client.get_collections()
    print("Qdrant connected, collections:", [c.name for c in collections.collections])
except Exception as e:
    print(f"Qdrant not available: {e}")