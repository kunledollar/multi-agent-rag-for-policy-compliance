# Sentinel
**Enterprise Policy & Compliance Intelligence Platform**

Sentinel is a production-grade, multi-agent RAG system designed to guide employees on enterprise policy and compliance in real time.

## Multi-Agent Architecture (Canonical)
1. Ingestion Agent — load docs, chunk, embed, store in FAISS  
2. Embedding Agent — convert text to embeddings (docs + queries)  
3. Retriever Agent — retrieve relevant chunks from FAISS  
4. Orchestrator Agent — controls execution flow across agents  
5. Answer Generation Agent — generates final response using retrieved context  
6. Reasoning Agent — produces structured justification/explanations  
7. Fact-Check Agent — verifies claims against retrieved sources  
8. Compliance Agent — determines compliance/violations/ambiguity  
9. RAGAS Evaluation Agent — evaluates relevance/faithfulness/context metrics  
10. Monitoring Agent — latency/errors/throughput metrics  
11. Tracing Agent — end-to-end request tracing and visibility  
12. API Gateway Agent — request validation, routing, error handling

## Quickstart (Local)
1) Copy `.env.example` → `.env` and set `OPENAI_API_KEY`  
2) Build + run:
```bash
docker compose up -d --build
```
3) Open:
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

## Notes
- `data/` holds policy documents.
- `artifacts/` stores FAISS index, embeddings, and RAGAS reports.
- `logs/` stores application logs.
