# 📄 RAG-Powered Intelligent Document Assistant

A generative-AI agent that answers questions over large, unstructured document sets (PDF, TXT, Markdown) using
**Retrieval-Augmented Generation**. Every answer is grounded in your documents, comes with **citations**, and the
assistant says *"I don't know"* instead of making things up.

Built with **Python · LangChain · LLaMA / GPT · ChromaDB (vector database) · FastAPI · Streamlit**.

---

## ✨ Features

- **Semantic search** - finds passages by meaning, not keywords (vector embeddings + cosine similarity)
- **Grounded answers with citations** - each answer lists the file, page and text snippets it was based on
- **Hallucination guardrails** - relevance gate + strict prompt + temperature 0 (see [below](#-how-hallucinations-are-reduced))
- **Swappable models** - local **LLaMA 3.1** through Ollama (free, private) *or* **GPT** through OpenAI; local HuggingFace or OpenAI embeddings
- **Three interfaces** - CLI, REST **API** (FastAPI + auto-generated Swagger docs), and a **Streamlit** chat UI
- **Persistent vector store** - index once, query any time; re-ingesting the same file never duplicates data
- **Evaluation harness** - measures retrieval hit-rate, answer accuracy and correct-refusal rate
- **Tested offline** - 10 unit tests run with no network, no API key and no model download
- **Docker-ready**

---

## 🧠 How it works

```
                ┌───────────────────────── INGESTION ─────────────────────────┐
 PDF/TXT/MD ──► │ Load ──► Chunk (800 chars, 150 overlap) ──► Embed ──► Chroma │
                └──────────────────────────────────────────────────────────────┘

                ┌───────────────────────────── QUERY ─────────────────────────────┐
 Question ──►   │ Embed ─► Top-k search ─► Relevance gate ─► Prompt+Context ─► LLM │ ──► Answer + Sources
                └───────────────────────────────┬─────────────────────────────────┘
                              no relevant chunk ─┘ → "I don't know" (LLM never called)
```

1. **Load** - PDFs are read page by page so citations include page numbers.
2. **Chunk** - `RecursiveCharacterTextSplitter` splits along paragraph / sentence boundaries with overlap, so each chunk is a coherent unit of meaning.
3. **Embed** - each chunk becomes a dense vector (`all-MiniLM-L6-v2`, 384 dims). Similar meaning ⇒ nearby vectors.
4. **Store** - vectors + metadata are saved in a persistent ChromaDB collection (cosine distance).
5. **Retrieve** - the question is embedded the same way; the `k` nearest chunks are fetched with relevance scores.
6. **Guard** - chunks below `MIN_RELEVANCE` are discarded. If nothing is left, the assistant declines without calling the LLM.
7. **Generate** - the surviving chunks are injected into a strict prompt; the LLM answers *only* from them and cites `[1]`, `[2]`…
8. **Return** - answer + the exact sources (file, page, score, snippet).

Full design rationale and tuning guide: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## 🗂 Project structure

```
rag-document-assistant/
├── rag_assistant/
│   ├── config.py         # settings from environment / .env
│   ├── loaders.py        # PDF / TXT / MD → LangChain Documents
│   ├── chunking.py       # recursive text chunking + deterministic chunk IDs
│   ├── embeddings.py     # HuggingFace or OpenAI embeddings
│   ├── vector_store.py   # ChromaDB (persistent, cosine)
│   ├── llm.py            # LLaMA (Ollama) or GPT (OpenAI) factory
│   ├── prompts.py        # grounding prompt (prompt engineering)
│   ├── pipeline.py       # RAGAssistant: ingest → retrieve → guard → generate → cite
│   ├── cli.py            # command-line interface
│   └── evaluation.py     # retrieval & answer-quality metrics
├── api/main.py           # FastAPI REST service
├── app.py                # Streamlit chat UI
├── data/sample_docs/     # demo documents (fictional company "Nimbus Labs")
├── eval/questions.json   # evaluation questions (in-scope + out-of-scope)
├── tests/                # offline unit + API tests
├── docs/ARCHITECTURE.md
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick start

### 1. Install
```bash
git clone https://github.com/<your-username>/rag-document-assistant.git
cd rag-document-assistant
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

### 2. Choose an LLM

**Option A - LLaMA locally (free, no API key)**
```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.1
# .env already defaults to LLM_PROVIDER=ollama
```

**Option B - GPT via OpenAI**
```env
# in .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

> The first run downloads the small embedding model (~90 MB) from HuggingFace automatically.

### 3. Index documents and ask questions
```bash
python -m rag_assistant.cli ingest data/sample_docs
python -m rag_assistant.cli ask "How many vacation days can be carried over?"
python -m rag_assistant.cli chat          # interactive
```

Example output:
```
Answer: Employees can carry over up to 8 unused vacation days into the next year [1].

  [1] employee_handbook.md  (relevance 0.71)
```

Out-of-scope question:
```
$ python -m rag_assistant.cli ask "Who won the FIFA World Cup in 2018?"
Answer: I don't know based on the provided documents.
```

Use your own files: `python -m rag_assistant.cli ingest path/to/your/folder`

### 4. Web UI
```bash
streamlit run app.py
```

### 5. REST API
```bash
uvicorn api.main:app --reload      # Swagger docs: http://localhost:8000/docs
```
```bash
# upload documents
curl -X POST http://localhost:8000/ingest -F "files=@report.pdf" -F "files=@notes.md"

# ask a question
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "What is the refund policy?", "top_k": 4}'
```
Response:
```json
{
  "question": "What is the refund policy?",
  "answer": "Monthly plans: 14 days. Annual plans: 30 days [1].",
  "grounded": true,
  "sources": [{"source": "product_faq.md", "page": null, "chunk_id": "…", "score": 0.68, "snippet": "…"}]
}
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | status + number of indexed chunks |
| `POST /ingest` | upload PDF/TXT/MD files |
| `POST /query` | ask a question |
| `DELETE /index` | wipe the vector store |

### 6. Docker
```bash
docker build -t rag-assistant .
docker run -p 8000:8000 --env-file .env rag-assistant
```
(When using Ollama from Docker, set `OLLAMA_BASE_URL=http://host.docker.internal:11434`.)

---

## 🛡 How hallucinations are reduced

| Layer | Mechanism |
|---|---|
| 1. Retrieval grounding | The LLM only sees retrieved chunks - it is never asked to answer from memory |
| 2. Relevance gate | Best chunk below `MIN_RELEVANCE` ⇒ LLM is **not called**, answer is "I don't know" |
| 3. Prompt engineering | "Use only the context, else reply with the refusal sentence, cite as [n]", `temperature=0` |
| 4. Verifiable output | Every answer ships with source file, page, score and snippet |

No RAG system can *guarantee* zero hallucination - a model can still misread a correct passage - so the evidence returned with each answer lets a human verify it quickly, and the evaluation harness lets you measure behaviour.

---

## 📊 Evaluation

```bash
python -m rag_assistant.cli ingest data/sample_docs
python -m rag_assistant.evaluation eval/questions.json
```

| Metric | Meaning |
|---|---|
| `retrieval_hit_rate` | fraction of questions where the expected source document was retrieved |
| `answer_keyword_rate` | fraction of answers containing the expected fact (e.g. "24") |
| `refusal_rate` | fraction of out-of-scope questions correctly refused |

Add your own cases to `eval/questions.json`:
```json
{"question": "…", "expected_source": "file.pdf", "expected_keywords": ["…"]}
{"question": "unrelated question"}
```

> 📝 After running it yourself, paste your real numbers here, e.g. *"With `all-MiniLM-L6-v2` + `llama3.1`: retrieval hit-rate X/9, refusal rate Y/3."*

---

## ⚙️ Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` (LLaMA) or `openai` (GPT) |
| `OLLAMA_MODEL` | `llama3.1` | any model you have pulled |
| `OPENAI_MODEL` | `gpt-4o-mini` | any OpenAI chat model |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` or `openai` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | chunking parameters (characters) |
| `TOP_K` | `4` | chunks retrieved per question |
| `MIN_RELEVANCE` | `0.30` | relevance threshold; below it the assistant refuses |
| `CHROMA_DIR` | `./chroma_db` | where the vector index is stored |

> ⚠️ If you change the embedding model, delete `chroma_db/` (or run `python -m rag_assistant.cli reset`) and re-ingest - vectors from different models are not compatible.

---

## 🧪 Tests

```bash
pytest -q
```
Tests use a deterministic hashing embedder and a fake LLM, so they need no internet, GPU or API key. They cover chunking, idempotent ingestion, retrieval accuracy, the relevance gate, refusal handling, the evaluation harness and the full REST flow.

---

## 🎯 Skills demonstrated

| Skill | Where |
|---|---|
| **Retrieval-Augmented Generation** | `pipeline.py` - end-to-end ingest → retrieve → generate |
| **Prompt engineering** | `prompts.py` - grounding rules, refusal contract, citation format |
| **Semantic search** | `embeddings.py`, `vector_store.py` - dense vectors + cosine similarity in ChromaDB |
| **API integration** | `llm.py` (OpenAI / Ollama), `api/main.py` (FastAPI service) |
| **Text chunking & vector embeddings** | `chunking.py`, `embeddings.py` |
| **Evaluation & testing** | `evaluation.py`, `tests/` |

---

## 🔮 Roadmap
Cross-encoder re-ranking · hybrid BM25 + vector search · conversation memory · streaming responses · OCR for scanned PDFs

## 📜 License
MIT - see [LICENSE](LICENSE).

**Author:** Aditya Sharma · IIT Ropar
