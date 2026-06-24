# Mini RAG Assistant

Готовий локальний RAG-проєкт для питань по відкритій технічній документації. Проєкт уже має підготовлений корпус документів, побудований Chroma index, фінальні evaluation results, Streamlit UI і FastAPI API.

LLM працює тільки локально через Ollama. API keys не потрібні.

## Поточний Стан

- Документи підготовлені в `data/raw/`.
- Chroma index побудований у `data/chroma/`.
- Chunk records збережені у `data/processed/chunks.jsonl`.
- Evaluation questions: `data/eval/eval_questions.json`.
- Evaluation results: `data/eval/eval_results.json`.
- Повний eval завершено: 20 questions x 5 retrieval modes = 100 runs.
- Streamlit UI: `http://localhost:8501`.
- FastAPI API: `http://localhost:8000`.

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
  "question": "How do I declare a path parameter?",
  "retrieval_mode": "hybrid",
  "metadata_filter": {}
}
```

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

За замовчуванням `EVAL_RESUME=true`, тому повторний запуск продовжує evaluation з уже збережених runs.

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
EVAL_RETRIEVAL_MODES=similarity,hybrid,metadata_filter,query_rewrite,rerank
EVAL_RETRIEVAL_ONLY=false
EVAL_RESUME=true
```

## Що Реалізовано

- Ingestion pipeline для вже підготовлених Markdown, HTML, TXT і text-based PDF документів.
- Chunking через LangChain `RecursiveCharacterTextSplitter`.
- Metadata для chunks: файл, URL, тип документа, джерело, PDF page number, section title, chunk id.
- Chroma vector index.
- Retrieval modes: `similarity`, `hybrid`, `metadata_filter`, `query_rewrite`, `rerank`.
- Answering тільки на основі retrieved context.
- Sources у відповіді.
- Honest fallback: `I don't know based on the provided context.`
- Streamlit demo.
- FastAPI endpoint.
- Evaluation pipeline з 20 питаннями і фінальними метриками.

## Фінальні Метрики

```text
Total questions: 20
Total runs: 100
Average latency: 13.285s
Source recall@k: 0.770
Groundedness score: 0.780
Answer keyword match score: 0.796
Best retrieval mode: metadata_filter
```

## Документація

- `report.md`: звіт по пунктах завдання і фінальні evaluation results.
- `data/eval/eval_questions.json`: 20 тестових питань.
- `data/eval/eval_results.json`: фінальні результати evaluation.

## Обмеження

- OCR немає, тому scanned/image-based PDF не підтримуються.
- Підтримуються тільки PDF з selectable text.
- LLM provider тільки Ollama.
- Швидкість залежить від локального комп'ютера і моделі Ollama.
- OpenAI/Gemini не використовувались, бо для стабільного повного evaluation вони потребували API keys, платних лімітів або підписки.
