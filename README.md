# Mini RAG Assistant

Готовий локальний RAG-проєкт для питань по відкритій технічній документації. Проєкт використовує LangChain, LangGraph, Chroma, локальні embeddings, Ollama, Streamlit UI і FastAPI API.

LLM працює локально через Ollama, тому OpenAI/Gemini API keys не потрібні.

## Поточний Стан

- Документи підготовлені в `data/raw/`.
- Chroma index побудований у `data/chroma/`.
- Chunk records збережені у `data/processed/chunks.jsonl`.
- Evaluation questions: `data/eval/eval_questions.json`.
- Evaluation results: `data/eval/eval_results.json`.
- Streamlit UI: `http://localhost:8501`.
- FastAPI API: `http://localhost:8000`.

## Retrieval Pipeline

У проєкті використовується один послідовний retrieval pipeline:

```text
question
-> LangGraph agent reads the question
-> agent calls search_rag_database(query, metadata)
-> retrieval pipeline: query rewrite -> metadata filter -> hybrid search -> rerank
-> agent reads tool output
-> answer with sources
```

Що це означає:

- `LangGraph agent`: LLM читає питання і сама генерує аргументи для tool;
- `search_rag_database`: tool викликає RAG pipeline і повертає chunks разом із source metadata;
- `metadata`: agent може сам передати `document_source`, `document_type`, `document_year`, `document_date`, `source_file` або `page_number`, якщо це є в питанні;
- `query rewriting`: retrieval pipeline переписує питання в коротший пошуковий запит;
- `metadata filter`: пошук обмежується потрібними документами, якщо metadata задана agent-ом або UI/API;
- `similarity search`: Chroma шукає chunks за embedding similarity;
- `BM25`: keyword search шукає chunks за точними словами;
- `fusion`: результати similarity і BM25 об'єднуються;
- `rerank`: знайдені candidates переоцінюються reranker-ом;
- `answering`: agent читає tool output, відповідає тільки на основі знайденого контексту і додає джерела.

## Запуск Streamlit UI

Переконайся, що Ollama встановлена і модель доступна:

```powershell
ollama pull llama3.1:8b
ollama run llama3.1:8b
```

Запуск UI:

```powershell
cd C:\Users\User\Documents\mini-rag-assistant
.\.venv\Scripts\python.exe -m streamlit run src/app/streamlit_app.py
```

Відкрити:

```text
http://localhost:8501
```

## Запуск FastAPI API

```powershell
cd C:\Users\User\Documents\mini-rag-assistant
.\.venv\Scripts\python.exe -m uvicorn src.app.api:app --reload --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

Основний endpoint:

```text
POST http://localhost:8000/ask
```

Приклад request:

```json
{
  "question": "Give me a FastAPI answer from 2022",
  "metadata_filter": {},
  "thread_id": "demo-thread"
}
```

Фільтр року можна передати явно:

```json
{
  "question": "What is FastAPI used for?",
  "metadata_filter": {
    "document_year": 2022
  }
}
```

Або написати рік прямо в питанні, наприклад `from 2022`. У такому випадку agent може сам передати `document_year=2022` у metadata tool call.

## Запуск Evaluation

Notebook:

```text
notebooks/run_eval.ipynb
```

PowerShell:

```powershell
cd C:\Users\User\Documents\mini-rag-assistant
.\.venv\Scripts\python.exe src/evaluation/run_eval.py
```

Результати записуються в:

```text
data/eval/eval_results.json
```

Останній повний запуск з LangGraph ReAct agent:

```text
Running 20 questions with full_pipeline and EVAL_SLEEP_SECONDS=0
Retrieval pipeline: full_pipeline
Pipeline steps: query_rewrite -> metadata_filter -> hybrid -> rerank
Total questions: 20
Total runs: 20
Average latency: 16.719s
Source recall@k: 0.850
Groundedness score: 0.673
Answer keyword match score: 0.804
```

## Основні Налаштування

```env
OLLAMA_MODEL=llama3.1:8b
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

VECTOR_DB=chroma
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=5
RERANK_TOP_N=3

RERANKER_PROVIDER=auto
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

EVAL_MAX_QUESTIONS=0
EVAL_SLEEP_SECONDS=0
EVAL_RESUME=FALSE
```

## Що Реалізовано

- Ingestion pipeline для підготовлених Markdown, HTML, TXT і text-based PDF документів.
- Chunking через LangChain `RecursiveCharacterTextSplitter`.
- Metadata для chunks: файл, URL, тип документа, джерело, document year, document date, PDF page number, section title, chunk id.
- Chroma vector index.
- Full retrieval pipeline: query rewrite, metadata filter, hybrid search, fusion, reranking.
- LangGraph ReAct agent через `langchain.agents.create_agent`.
- RAG tool `search_rag_database(query, metadata)`.
- Metadata schema `DocumentMetadata` для agent-generated filters.
- Answering тільки на основі tool output.
- In-memory checkpointer memory через `MemorySaver`.
- Sources у відповіді.
- Honest fallback: `I don't know based on the provided context.`
- Streamlit demo.
- FastAPI endpoint.
- Evaluation pipeline з 20 питаннями і метриками.

## Документація

- `report.md`: звіт по пунктах завдання, метрики і приклади.
- `data/eval/eval_questions.json`: 20 тестових питань.
- `data/eval/eval_results.json`: результати evaluation.

## Обмеження

- OCR немає, тому scanned/image-based PDF не підтримуються.
- Підтримуються тільки PDF з selectable text.
- LLM provider тільки Ollama.
- Швидкість залежить від локального комп'ютера і моделі Ollama.
- OpenAI/Gemini не використовувались, бо для стабільного повного evaluation вони потребували API keys, платних лімітів або підписки.
