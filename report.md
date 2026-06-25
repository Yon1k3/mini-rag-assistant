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

У проєкті використовується LangGraph ReAct agent і один послідовний retrieval pipeline:

```text
question
-> LangGraph agent reads the question
-> agent calls search_rag_database(query, metadata)
-> retrieval pipeline: query rewrite -> metadata filter -> hybrid search -> rerank
-> agent reads tool output
-> answer with sources
```

Як це реалізовано:

- `src/answering/rag_chain.py`: створює `RAGAgent` через `langchain.agents.create_agent`.
- `search_rag_database`: tool, який приймає `query` і optional `DocumentMetadata`.
- `DocumentMetadata`: schema для metadata filters: `document_source`, `document_type`, `document_year`, `document_date`, `source_file`, `page_number`.
- `query_rewriter.py`: Ollama переписує питання в коротший search query.
- `retriever.py`: підключає embeddings, Chroma і запускає повний pipeline.
- `hybrid_retriever.py`: виконує dense similarity search + BM25 keyword search.
- `reciprocal_rank_fusion`: об'єднує результати vector search і keyword search.
- `reranker.py`: переоцінює candidates через cross-encoder або lexical fallback.
- Metadata filter застосовується перед пошуком, якщо agent витягнув metadata із питання або якщо користувач передав filter через UI/API.

Такий workflow дозволяє LLM самостійно сформувати tool arguments, викликати RAG і потім відповісти на основі tool output.

## 5. Answering

Вимога: відповідати тільки на основі знайденого контексту, додавати джерела і чесно казати "не знаю", якщо контексту недостатньо.

Реалізація:

- RAG pipeline: `src/answering/rag_chain.py`.
- `RAGAgent` створює agent через `create_agent`.
- `search_rag_database` повертає chunks у текстовому форматі з `source_file`, `document_source`, `document_year`, `page_number`, `chunk_id`, `url` і `Content`.
- `MemorySaver` використовується як in-memory checkpointer для діалогової пам'яті.
- `thread_id` відокремлює різні діалоги в Streamlit, FastAPI і evaluation.
- System prompt вимагає завжди викликати tool, не використовувати зовнішні знання і цитувати джерела.
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

Evaluation рахується як:

```text
20 questions x 1 full pipeline = 20 runs
```

Останній запуск з LangGraph ReAct agent:

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
- LangGraph ReAct agent з RAG tool і `MemorySaver`.
- Agent-generated metadata filters через `DocumentMetadata`.
- Відповідь тільки на основі tool output.
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

## 10. Приклади Відповідей

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

## 11. Висновок

Проєкт закриває основні вимоги mini RAG assistant: ingestion, chunking, metadata, vector index, покращений retrieval pipeline, answering із sources, чесне "не знаю", evaluation pipeline, Streamlit UI і FastAPI API. Поточна answering логіка працює через LangGraph ReAct agent: LLM генерує аргументи для RAG tool, tool повертає chunks із source metadata, а фінальна відповідь формується тільки на основі цього tool output.
