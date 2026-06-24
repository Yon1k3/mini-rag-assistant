# Звіт: Mini RAG Assistant

## 1. Мета Проєкту

Мета завдання - зробити міні RAG асистент для відкритої технічної документації. Система має працювати з приблизно 50 документами, будувати векторний індекс, знаходити релевантний контекст, відповідати тільки на основі цього контексту, показувати джерела і мати evaluation pipeline.

Проєкт реалізовано на Python з використанням LangChain, Chroma, локальних Hugging Face embeddings, Ollama, Streamlit і FastAPI.

## 2. Вхідні Документи

У проєкті підготовлено 60 документів з відкритої технічної документації. Підтримані формати:

- Markdown;
- HTML;
- TXT;
- text-based PDF.

## 3. Ingestion Pipeline

Вимога: підготувати документи, порізати їх на chunks, зберегти metadata і побудувати vector index.

Реалізація:

- `src/ingestion/loaders.py` читає Markdown, HTML, TXT і PDF.
- `src/ingestion/chunking.py` ріже документи через LangChain `RecursiveCharacterTextSplitter`.
- `src/ingestion/metadata.py` нормалізує metadata.
- `src/ingestion/build_index.py` будує Chroma index для підготовлених документів.
- Chunks збережені у `data/processed/chunks.jsonl`.
- Vector index збережений у `data/chroma/`.

Metadata для chunks:

- `source_file`;
- `source_url`;
- `document_source`;
- `document_type`;
- `page_number` для PDF;
- `section_title`;
- `chunk_id`;
- `ingestion_timestamp`.

Поточний індекс містить 3482 chunks.

## 4. Retrieval Step

Вимога: зробити similarity search і хоча б одну покращену retrieval стратегію.

Реалізація містить 5 retrieval modes:

- `similarity`: dense vector search у Chroma;
- `hybrid`: Chroma search + BM25 keyword search з Reciprocal Rank Fusion;
- `metadata_filter`: пошук з фільтром по metadata;
- `query_rewrite`: Ollama переписує питання у коротший search query;
- `rerank`: candidate chunks додатково ранжуються reranker-ом.

Основний код retrieval знаходиться у `src/retrieval/`.

## 5. Answering

Вимога: відповідати тільки на основі знайденого контексту, додавати джерела і чесно казати "не знаю", якщо контексту недостатньо.

Реалізація:

- RAG pipeline: `src/answering/rag_chain.py`.
- Prompt: `src/answering/prompts.py`.
- LLM: Ollama через `langchain-ollama`.
- Sources формуються кодом з metadata retrieved chunks.
- Якщо контекст не знайдено, система повертає:

```text
I don't know based on the provided context.
```

## 6. Evaluation

Вимога: створити 20 тестових питань, вказати expected sources і expected answer, порахувати прості метрики.

Реалізація:

- Questions: `data/eval/eval_questions.json`.
- Results: `data/eval/eval_results.json`.
- Script: `src/evaluation/run_eval.py`.
- Notebook: `notebooks/run_eval.ipynb`.

Метрики:

- `source_recall_at_k`: чи правильне джерело знайдено у top-k;
- `groundedness`: наскільки відповідь спирається на retrieved context;
- `answer_keyword_match_score`: чи є очікувані ключові слова;
- `latency_seconds`: час відповіді;
- `best_retrieval_mode`: найкращий retrieval режим за composite score.

Повний evaluation завершено:

```text
status = complete
total_questions = 20
total_runs = 100
```

Фінальні результати:

- Average latency: 13.285s;
- Source recall@k: 0.770;
- Groundedness score: 0.780;
- Answer keyword match score: 0.796;
- Best retrieval mode: `metadata_filter`.

## 7. Demo

У проєкті є два способи взаємодії:

- Streamlit UI: `src/app/streamlit_app.py`;
- FastAPI API: `src/app/api.py`.

FastAPI endpoint:

- `POST /ask`;
- Swagger UI: `http://localhost:8000/docs`.

## 8. 5 Хороших Прикладів

1. `q001` / `metadata_filter`: правильне джерело знайдено, groundedness = 1.0, keyword score = 1.0.
2. `q001` / `rerank`: правильне джерело знайдено, groundedness = 1.0, keyword score = 1.0.
3. `q003` / `hybrid`: правильне джерело знайдено, groundedness = 1.0, keyword score = 1.0.
4. `q016` / `metadata_filter`: правильне джерело знайдено, groundedness = 1.0, keyword score = 1.0.
5. `q005` / `metadata_filter`: правильне джерело знайдено, groundedness = 1.0, відповідь релевантна, keyword score = 0.667.

## 9. 5 Слабких Або Failed Прикладів

Повних technical failures не було, але були слабші приклади за метриками:

1. `q002` / `query_rewrite`: expected source не знайдено, groundedness = 0.714.
2. `q016` / `query_rewrite`: expected source не знайдено, keyword score = 0.667.
3. `q019` / `query_rewrite`: expected source не знайдено, groundedness = 0.542.
4. `q002` / `rerank`: expected source не знайдено, groundedness = 0.515.
5. `q013` / `query_rewrite`: expected source не знайдено, groundedness = 0.222, keyword score = 0.667.

## 10. Обмеження

- Немає OCR для scanned PDF.
- LLM provider тільки Ollama.
- Якість залежить від локальної Ollama model.
- Повний eval повільний, бо відповіді генеруються локально.
- Groundedness metric евристичний і не замінює ручну перевірку.

## 11. Висновок

Проєкт закриває основні вимоги mini RAG assistant: ingestion, chunking, metadata, vector index, кілька retrieval стратегій, answering із sources, чесне "не знаю", evaluation pipeline, Streamlit UI і FastAPI API. Фінальний evaluation показав нормальний результат для навчального локального RAG проєкту: source recall@k 0.770, groundedness 0.780 і keyword match 0.796.
