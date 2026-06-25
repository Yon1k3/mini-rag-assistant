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
-> rewrite query
-> apply metadata filter
-> similarity search + BM25
-> fusion
-> rerank
-> answer with sources
```

Що це означає:

- `query rewriting`: Ollama переписує питання в коротший пошуковий запит;
- `metadata filter`: якщо користувач задав фільтр або в питанні є рік, пошук обмежується потрібними документами;
- `similarity search`: Chroma шукає chunks за embedding similarity;
- `BM25`: keyword search шукає chunks за точними словами;
- `fusion`: результати similarity і BM25 об'єднуються;
- `rerank`: знайдені candidates переоцінюються reranker-ом;
- `answering`: LLM відповідає тільки на основі знайденого контексту і додає джерела.

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

Або написати рік прямо в питанні, наприклад `from 2022`. У такому випадку pipeline автоматично застосує `document_year=2022`.

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

Останній повний запуск після об'єднання retrieval стратегій і переходу на LangGraph:

```text
Running 20 questions with full_pipeline and EVAL_SLEEP_SECONDS=0
Retrieval pipeline: full_pipeline
Pipeline steps: query_rewrite -> metadata_filter -> hybrid -> rerank
Total questions: 20
Total runs: 20
Average latency: 13.765s
Source recall@k: 0.900
Groundedness score: 0.762
Answer keyword match score: 0.800
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
- Answering тільки на основі retrieved context.
- LangGraph answering graph зі state, nodes і in-memory checkpointer memory.
- Sources у відповіді.
- Honest fallback: `I don't know based on the provided context.`
- Streamlit demo.
- FastAPI endpoint.
- Evaluation pipeline з 20 питаннями і метриками.

## Git Гілки Та Зміни

### `add_generate_docs`

У цій гілці було додано metadata для документів і chunks:

- створено `data/processed/source_metadata.json`;
- додано `src/ingestion/metadata.py` для нормалізації metadata;
- додано поля `document_year` і `document_date`;
- ingestion pipeline почав переносити ці поля у chunks і Chroma index;
- локально перебудовано vector database, щоб пошук міг використовувати metadata filtering.

Роки в `document_year` використовуються як демонстраційна metadata для фільтрації, наприклад запитів типу `from 2022`.

### `add_combine_retrievers`

У цій гілці окремі retrieval режими були об'єднані в один послідовний pipeline:

```text
query_rewrite -> metadata_filter -> hybrid -> rerank
```

Що змінилось:

- `src/retrieval/retriever.py` запускає повний retrieval pipeline;
- query rewriting готує кращий search query;
- metadata filtering застосовується перед пошуком;
- hybrid search поєднує Chroma similarity search і BM25;
- fusion об'єднує dense і keyword результати;
- reranker переоцінює знайдені candidates;
- `src/evaluation/run_eval.py` рахує evaluation для одного full pipeline, а не для п'яти окремих режимів.

### `refactor_to_langgraph`

У цій гілці answering частина була перероблена на LangGraph:

- `src/answering/rag_chain.py` використовує `StateGraph`;
- додано state для питання, retrieved chunks, context, answer, sources і messages;
- answering розділено на nodes: `initialize_state`, `retrieve_context`, `prepare_context`, `generate_answer`, `finalize_response`;
- додано `MemorySaver` checkpointer;
- додано `thread_id`, щоб Streamlit і FastAPI могли підтримувати окремі діалоги;
- evaluation передає окремий `thread_id` для кожного питання, щоб пам'ять не змішувала тестові запити.

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
