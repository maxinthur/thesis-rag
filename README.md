# Thesis RAG — Retrieval-Augmented Q&A over a PhD Thesis

A RAG system for querying *Reconstructing Neural Dynamics Underlying Cognitive Flexibility Using Parameter-Evolving RNNs* (Max Thurm, Heidelberg University, 2025).

The focus here isn't just making something that returns plausible answers — it's building a system where you can actually tell *why* it fails when it does.

> *"Good fit does not guarantee correct mechanism."* — A lesson from the thesis, applied to the system that retrieves it.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────┐     ┌──────────────────┐
│  Voyage AI      │───▶ │   ChromaDB       │
│  voyage-3-large │     │   (HNSW, cosine) │
└─────────────────┘     └────────┬─────────┘
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

**Stack:** Python · Streamlit · ChromaDB · Anthropic Claude · Voyage AI · Docker · Azure Container Apps

Embeddings are asymmetric: `input_type="document"` at ingest time, `input_type="query"` at retrieval time. This matters more than it sounds — switching from a symmetric 384d model to Voyage's 1024d asymmetric embeddings pushed retrieval relevance from 0.59 to 0.87 in eval.

---

## Evaluation Framework

The system has a four-criterion evaluation pipeline, loosely inspired by the thesis's own validation methodology. The point is that a single aggregate score hides where things break.

| Criterion | What it measures | Failure it catches |
|---|---|---|
| **Retrieval Relevance** | Are the retrieved chunks actually relevant? | Bad chunking, embedding mismatch |
| **Faithfulness** | Is the answer grounded in the context? | Hallucination, context leakage |
| **Answer Completeness** | Does the answer cover expected key points? | Insufficient retrieval, shallow generation |
| **Hallucination-Free** | Are there claims beyond the context? | Confabulation under uncertainty |

```bash
python evaluate.py --verbose
```

Results go to `eval_results.json` with per-question scores and per-category breakdowns. The main finding from running this: faithfulness was >0.96 across all configurations — the LLM wasn't the problem. Retrieval was. One question (q10, RNN vs. neural recordings) still fails across all configs because the answer uses completely different terminology than the question. That's the argument for hybrid search.

---

## Quick Start (Local)

```bash
git clone https://github.com/maxinthur/thesis-rag.git
cd thesis-rag

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set keys
export ANTHROPIC_API_KEY="sk-ant-..."
export VOYAGE_API_KEY="pa-..."

python ingest.py --pdf data/thesis.pdf
python -m streamlit run app.py
```

### Docker

```bash
docker build -t thesis-rag .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e VOYAGE_API_KEY=pa-... \
  thesis-rag
```

---

## Deploy to Azure Container Apps

### Prerequisites
- Azure account
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
- Docker

### Steps

```bash
az login

# One-time infra setup
az group create --name thesis-rag-rg --location spaincentral
az acr create --name thesisragcr --resource-group thesis-rag-rg --sku Basic --admin-enabled true
az containerapp env create --name thesis-rag-env --resource-group thesis-rag-rg --location spaincentral

# Build and push (ACR Tasks are disabled on free tier, so build locally)
az acr login --name thesisragcr
docker build -t thesisragcr.azurecr.io/thesis-rag:latest .
docker push thesisragcr.azurecr.io/thesis-rag:latest

# Deploy
az containerapp create \
  --name thesis-rag \
  --resource-group thesis-rag-rg \
  --environment thesis-rag-env \
  --image thesisragcr.azurecr.io/thesis-rag:latest \
  --target-port 8000 \
  --ingress external \
  --registry-server thesisragcr.azurecr.io \
  --secrets anthropic-key=$ANTHROPIC_API_KEY voyage-key=$VOYAGE_API_KEY app-pw=thesis2025 \
  --env-vars ANTHROPIC_API_KEY=secretref:anthropic-key VOYAGE_API_KEY=secretref:voyage-key APP_PASSWORD=secretref:app-pw

# Get URL
az containerapp show \
  --name thesis-rag \
  --resource-group thesis-rag-rg \
  --query properties.configuration.ingress.fqdn \
  --output tsv
```

Subsequent deploys: push a new image and run `az containerapp update --image ...`. The GitHub Actions workflow handles this automatically on push to main.

### Teardown
```bash
az group delete --name thesis-rag-rg --yes --no-wait
```

---

## Project Structure

```
thesis-rag/
├── app.py              ← Streamlit UI + auth + rate limiting
├── ingest.py           ← PDF → chunks → Voyage embeddings → ChromaDB
├── retrieval.py        ← Semantic search + Claude answer generation
├── evaluate.py         ← Four-criterion evaluation pipeline
├── analyze.py          ← PDF statistics + chunking comparison
├── config.py           ← Central config (models, paths, prompts)
├── eval_questions.json ← 10 ground-truth Q&A pairs across 4 categories
├── Dockerfile
├── requirements.txt
└── data/
    └── thesis.pdf
```

---

## Design Decisions

**Why Voyage AI over sentence-transformers?** The evaluation made this clear — retrieval relevance jumped from 0.59 to 0.84 just by switching embedding models. `voyage-3-large` also supports asymmetric embeddings natively, which matters for document retrieval. The tradeoff is an extra API dependency and cost, but for a document this size it's negligible.
---

## What's Missing
- Persistent vector store — currently the ChromaDB index is baked into the container image, which works but is inelegant
- Multi-document support
---

## Author

**Max Thurm** — PhD in Computational Neuroscience, Heidelberg University.  
Working on dynamical systems reconstruction, neural population geometry, and model validation under non-stationarity.