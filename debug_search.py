import sys
sys.path.insert(0, '.')
from app.rag.vector_store import search

candidates = search("How to troubleshoot low airflow in AHU-02?", top_k=15)
print(f"Number of candidates: {len(candidates)}")
for i, (text, score) in enumerate(candidates[:5]):
    print(f"  {i}: score={score}, text={text[:80]}...")

# Check if candidates have text
if candidates:
    print(f"\nFirst candidate text type: {type(candidates[0][0])}")
    print(f"First candidate text: {candidates[0][0][:100]}")