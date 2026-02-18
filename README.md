# Thesis RAG — Retrieval-Augmented Q&A over a PhD Thesis

A RAG system that answers questions about *Reconstructing Neural Dynamics Underlying Cognitive Flexibility Using Parameter-Evolving RNNs* (Max Thurm, Heidelberg University, 2025).

Built to demonstrate not just RAG implementation, but **systematic evaluation** — because a model that gives plausible answers is not the same as a model that gives correct ones.

> *"Good fit does not guarantee correct mechanism."* — A lesson from the thesis, applied to the system that retrieves it.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────────┐
│  Embedding  │───▶│   ChromaDB       │
│  (MiniLM)   │     │   (HNSW, cosine) │
└─────────────┘     └────────┬─────────┘
                             │ top-k chunks
                             ▼
                    ┌──────────────────┐
                    │  Claude (Sonnet) │
                    │  + System Prompt │
                    └────────┬─────────┘
                             │
                             ▼
                       Answer + Sources
```

**Stack:** Python · Streamlit · ChromaDB · Anthropic Claude · Sentence Transformers · Docker · Azure Container Apps

---

## Evaluation Framework

The system includes a **four-criterion evaluation pipeline** inspired by the thesis's own validation methodology — multiple orthogonal metrics that diagnose specific failure modes:

| Criterion | What it measures | Failure it catches |
|---|---|---|
| **Retrieval Relevance** | Are the retrieved chunks actually relevant? | Bad chunking, embedding mismatch |
| **Faithfulness** | Is the answer grounded in the context? | Hallucination, context leakage |
| **Answer Completeness** | Does the answer cover expected key points? | Insufficient retrieval, shallow generation |
| **Hallucination-Free** | Are there claims beyond the context? | Confabulation under uncertainty |

Run the evaluation:
```bash
python evaluate.py --verbose
```

This produces per-question scores, per-category breakdowns (architecture / methodology / results / validation), and aggregate metrics — saved to `eval_results.json`.

---

## Quick Start (Local)

```bash
# Clone and enter
git clone https://github.com/maxthurm/thesis-rag.git
cd thesis-rag

# Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set API key
cp .env.example .env
# Edit .env with your Anthropic API key

# Ingest thesis PDF
python ingest.py --pdf data/thesis.pdf

# Run the app
streamlit run app.py
```

### Docker (Local)

```bash
docker build -t thesis-rag .
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-ant-... thesis-rag
```

---

## Deploy to Azure Container Apps

### Prerequisites
- Azure account (Free Tier works)
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed

### Steps

```bash
# 1. Login
az login

# 2. Create resource group
az group create --name thesis-rag-rg --location germanywestcentral

# 3. Create Container Apps environment
az containerapp env create \
  --name thesis-rag-env \
  --resource-group thesis-rag-rg \
  --location germanywestcentral

# 4. Build and deploy (Azure builds the Docker image for you)
az containerapp up \
  --name thesis-rag \
  --resource-group thesis-rag-rg \
  --environment thesis-rag-env \
  --source . \
  --target-port 8080 \
  --ingress external \
  --env-vars ANTHROPIC_API_KEY=<your-key>

# 5. Get the URL
az containerapp show \
  --name thesis-rag \
  --resource-group thesis-rag-rg \
  --query properties.configuration.ingress.fqdn \
  --output tsv
```

Your app is now live at `https://thesis-rag.<hash>.germanywestcentral.azurecontainerapps.io`

### Teardown (avoid charges)
```bash
az group delete --name thesis-rag-rg --yes --no-wait
```

---

## Project Structure

```
thesis-rag/
├── app.py              ← Streamlit UI
├── ingest.py           ← PDF → chunks → embeddings → ChromaDB
├── retrieval.py        ← Semantic search + Claude answer generation
├── evaluate.py         ← Multi-criterion evaluation pipeline
├── config.py           ← Central configuration
├── eval_questions.json ← Ground-truth Q&A pairs
├── Dockerfile          ← Container definition
├── requirements.txt
├── .env.example
└── data/
    └── thesis.pdf      ← Your PDF goes here
```

---

## Design Decisions

**Why ChromaDB instead of pgvector?** For a self-contained demo, an embedded vector store avoids external database dependencies. In production, I would use PostgreSQL with pgvector for hybrid search (vector + full-text via RRF) and row-level security — see [this excellent overview](https://scieneers.de/rag-mit-postgresql/) of the approach.

**Why a local embedding model?** `all-MiniLM-L6-v2` runs inside the container with no API calls, reducing latency and cost. For production, a larger model or API-based embeddings (e.g., `text-embedding-3-small`) would improve retrieval quality.

**Why Claude for evaluation?** LLM-as-judge is the pragmatic standard for RAG evaluation. The four criteria are designed to be orthogonal — each diagnoses a different failure mode, mirroring the multi-criterion validation philosophy from the thesis itself.

---

## Future Improvements

- **Hybrid search:** Add BM25/full-text search and fuse via Reciprocal Rank Fusion
- **pgvector migration:** Replace ChromaDB with PostgreSQL for production-grade storage
- **Chunking experiments:** Compare fixed-size vs. semantic chunking strategies
- **Multi-document support:** Extend to related publications
- **LiteLLM integration:** Unified API layer for provider-agnostic deployment

---

## Author

**Max Thurm** — PhD in Computational Neuroscience, Heidelberg University.
Specializing in machine learning, dynamical systems reconstruction, and model validation under non-stationarity.

---

*Hinweis / Note:* Dieses Projekt wurde als Proof-of-Concept für RAG-Architekturen entwickelt. Es demonstriert den vollständigen Workflow von Dokumentenaufbereitung über Retrieval bis zur systematischen Evaluation — mit besonderem Fokus auf Validierungsmethodik, die über Standard-Accuracy-Metriken hinausgeht. / This project was built as a RAG architecture proof-of-concept, demonstrating the full pipeline from document ingestion through retrieval to systematic evaluation — with a focus on validation methodology that goes beyond standard accuracy metrics.
