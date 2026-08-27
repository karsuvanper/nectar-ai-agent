from app.rag.retriever import query_rag_agent

# Test query as specified in the instructions
result = query_rag_agent("How to troubleshoot low airflow in AHU-02?")
print("Query result:")
print(result)