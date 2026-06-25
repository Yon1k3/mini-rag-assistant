# Звіт: Mini RAG Assistant

## 1. Мета Проєкту

Мета завдання - зробити міні RAG асистент для відкритої технічної документації. Система працює з підготовленим набором документів, будує vector index, знаходить релевантний контекст, відповідає тільки на основі цього контексту, показує джерела і має evaluation pipeline.

Проєкт реалізовано на Python з використанням LangChain, LangGraph, Chroma, локальних Hugging Face embeddings, Ollama, Streamlit і FastAPI.

## 2. Вхідні Документи

У проєкті підготовлено корпус з відкритої технічної документації. Використані документи по FastAPI, Pydantic, LangChain і Python, бо вони добре підходять для RAG: мають структурований текст, API-приклади, багато точних технічних термінів і зрозумілі джерела.

Підтримані формати:

- Markdown;
- HTML;
- TXT;
- text-based PDF.

## 3. Ingestion Pipeline

Вимога: завантажити документи, порізати їх на chunks, зберегти metadata і побудувати vector index.

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
- `document_year`;
- `document_date`;
- `page_number` для PDF;
- `section_title`;
- `chunk_id`;
- `ingestion_timestamp`.

## 4. Retrieval Step

Вимога: зробити similarity search і хоча б одну покращену retrieval стратегію.

У фінальній версії використовується не набір окремих режимів, а один послідовний pipeline:

```text
question
-> rewrite query
-> apply metadata filter
-> similarity search + BM25
-> fusion
-> rerank
-> answer with sources
```

Як це реалізовано:

- `query_rewriter.py`: Ollama переписує питання в коротший search query.
- `retriever.py`: підключає embeddings, Chroma і запускає повний pipeline.
- `hybrid_retriever.py`: виконує dense similarity search + BM25 keyword search.
- `reciprocal_rank_fusion`: об'єднує результати vector search і keyword search.
- `reranker.py`: переоцінює candidates через cross-encoder або lexical fallback.
- Metadata filter застосовується перед пошуком, якщо користувач передав фільтр або якщо система знайшла рік у питанні, наприклад `2022`.

Такий pipeline використовує всі покращені стратегії послідовно, а не окремо.

## 5. Answering

Вимога: відповідати тільки на основі знайденого контексту, додавати джерела і чесно казати "не знаю", якщо контексту недостатньо.

Реалізація:

- RAG pipeline: `src/answering/rag_chain.py`.
- Answering перероблено на LangGraph graph зі state, окремими nodes і `MemorySaver` checkpointer memory.
- Основні nodes: `initialize_state`, `retrieve_context`, `prepare_context`, `generate_answer`, `finalize_response`.
- Prompt: `src/answering/prompts.py`.
- LLM: Ollama через `langchain-ollama`.
- Sources формуються з metadata retrieved chunks.
- Якщо контекст не знайдено або LLM не може відповісти, система повертає:

```text
I don't know based on the provided context.
```

## 6. Evaluation

Вимога: створити 20 тестових питань, очікувані джерела, очікувану відповідь і порахувати прості метрики.

Реалізація:

- Questions: `data/eval/eval_questions.json`.
- Results: `data/eval/eval_results.json`.
- Script: `src/evaluation/run_eval.py`.
- Notebook: `notebooks/run_eval.ipynb`.

Метрики:

- `source_recall_at_k`: чи правильне джерело знайдено у top-k;
- `groundedness`: наскільки відповідь спирається на retrieved context;
- `answer_keyword_match_score`: чи є очікувані ключові слова;
- `latency_seconds`: час відповіді.

Після переходу на один full pipeline evaluation рахується як:

```text
20 questions x 1 full pipeline = 20 runs
```

Останній запуск після об'єднання retrieval стратегій і переходу на LangGraph:

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

## 7. Demo

У проєкті є два способи взаємодії:

- Streamlit UI: `src/app/streamlit_app.py`;
- FastAPI API: `src/app/api.py`.

FastAPI endpoint:

- `POST /ask`;
- Swagger UI: `http://localhost:8000/docs`.

Приклад request:

```json
{
  "question": "Give me a FastAPI answer from 2022",
  "metadata_filter": {
    "document_year": 2022
  },
  "thread_id": "demo-thread"
}
```

## 8. Що Працює

- Підготовка документів і chunking.
- Metadata для кожного chunk.
- Chroma vector index.
- Full retrieval pipeline: query rewrite, metadata filter, hybrid search, fusion, rerank.
- LangGraph state graph для answering з memory/checkpointer.
- Відповідь тільки на основі retrieved context.
- Sources у відповіді.
- Streamlit UI.
- FastAPI endpoint.
- Evaluation script і notebook.

## 9. Що Не Працює Або Має Обмеження

- OCR немає, тому scanned/image-based PDF не підтримуються.
- Підтримуються тільки PDF з selectable text.
- LLM provider тільки Ollama.
- Якість і швидкість залежать від локальної Ollama model і потужності комп'ютера.
- Якщо Ollama server не запущений, query rewriting і answering повернуть помилку підключення.
- Groundedness metric евристичний і не замінює ручну перевірку.

## 10. Git Гілки Та Зміни

### `add_generate_docs`

Ця гілка відповідала за metadata і підготовку документів до фільтрації:

- додано `data/processed/source_metadata.json`;
- додано `src/ingestion/metadata.py`;
- у metadata з'явились поля `document_year` і `document_date`;
- ingestion pipeline почав зберігати ці поля в chunks;
- Chroma index було перебудовано локально, щоб metadata потрапила у vector database.

Роки в metadata використовуються для демонстрації metadata filtering. Наприклад, FastAPI має `document_year=2022`, тому запит з `from 2022` або явний filter `{"document_year": 2022}` обмежує пошук відповідними chunks.

### `add_combine_retrievers`

Ця гілка змінила retrieval logic. До цього різні стратегії можна було порівнювати окремо. Після зміни вони працюють як один послідовний pipeline:

```text
question
-> rewrite query
-> apply metadata filter
-> similarity search + BM25
-> fusion
-> rerank
-> answer with sources
```

Основні зміни:

- `src/retrieval/retriever.py` став головною точкою запуску full pipeline;
- query rewriting переписує питання в коротший пошуковий запит;
- metadata filtering застосовується до dense і keyword пошуку;
- hybrid search поєднує Chroma similarity search і BM25;
- fusion об'єднує результати двох пошуків;
- reranker залишає найкращі chunks для відповіді;
- `src/evaluation/run_eval.py` рахує метрики для одного `full_pipeline`.

### `refactor_to_langgraph`

Ця гілка змінила answering layer без зміни зовнішньої логіки відповіді:

- `src/answering/rag_chain.py` перероблено на LangGraph `StateGraph`;
- додано state для question, metadata filter, retrieved chunks, context, answer, sources і messages;
- pipeline answering розділено на nodes: `initialize_state`, `retrieve_context`, `prepare_context`, `generate_answer`, `finalize_response`;
- додано `MemorySaver` checkpointer memory;
- додано `thread_id` для окремих діалогів у Streamlit, FastAPI і evaluation;
- `src/app/api.py` і `src/app/streamlit_app.py` передають `thread_id`;
- `src/evaluation/run_eval.py` створює окремий `thread_id` для кожного eval question.

## 11. Приклади Відповідей

5 хороших прикладів з останнього evaluation:

- `q001`: What is FastAPI used for?
- `q016`: What does the Python tutorial cover?
- `q005`: What are dependencies in FastAPI?
- `q009`: What does Pydantic serialization do?
- `q006`: What is a Pydantic model?

5 слабких прикладів з останнього evaluation:

- `q018`: low metric score;
- `q008`: low metric score;
- `q012`: low metric score;
- `q002`: low metric score;
- `q017`: low metric score.

## 12. Висновок

Проєкт закриває основні вимоги mini RAG assistant: ingestion, chunking, metadata, vector index, покращений retrieval pipeline, answering із sources, чесне "не знаю", evaluation pipeline, Streamlit UI і FastAPI API. Фінальна retrieval логіка працює послідовно: спочатку query rewriting, потім metadata filtering, hybrid search, fusion, reranking і тільки після цього generation відповіді.
