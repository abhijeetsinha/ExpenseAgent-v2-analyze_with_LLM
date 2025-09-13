
# Expense Analysis Agent (Azure OpenAI + Chroma + Pandas)

A **beginner‑friendly** but **production‑sensible** Python agent that answers *any* expense question using a hybrid approach:

*   **Analytics questions** (totals, averages, breakdowns, time windows):  
    The LLM produces a small **JSON plan** (filters, group\_by, aggregations, etc.).  
    A tiny **deterministic executor** runs that plan over **all rows** using **pandas**.  
    → **Correct numbers** without relying on top‑k retrieval.

*   **Lookup / descriptive questions** (find similar entries, show examples, explain patterns):  
    Uses **semantic retrieval** from a local **Chroma** vector database and lets the LLM **explain**.

This design avoids the common pitfall of using top‑k vector hits for sums or group‑bys (which is inaccurate by design).

***

## Why this architecture?

> **Top‑k retrieval is not a database.**  
> If you compute totals/averages/group‑bys from a handful of “nearest” rows, you’ll miss data and get wrong answers.  
> The fix is to **separate “what to do” from “how to do it.”**

*   The **LLM plans** the operation (what to filter, group by, and aggregate).
*   **Pandas executes** the plan on the **full normalized table** (not a subset).
*   For open‑ended questions, **RAG** (Chroma) gives the LLM just enough relevant context to respond clearly.

***

## Features

*   Ingest multiple **CSV / XLSX / XLS** files from a folder
*   Row‑level **embeddings** with **Azure OpenAI**
*   Local **vector store** with **Chroma** (`./chroma_db`)
*   Persisted **normalized table** for exact analytics (`./store/expenses.parquet`)
*   **Query planner** (LLM → JSON) + **tiny analytics executor** (pandas)
*   **Fallback RAG** path for non‑analytics questions
*   Beginner‑friendly code, extensive console diagnostics

***

## Project structure

    .
    ├─ app.py                     # main CLI app (ingest + ask)
    ├─ .env                       # Azure OpenAI credentials & model names
    ├─ data/                      # put your expense files here
    ├─ chroma_db/                 # Chroma vector store (auto-created)
    ├─ store/expenses.parquet     # normalized table (auto-created after ingest)
    └─ requirements.txt           # dependencies (see below)

***

## Setup

### 1) Python & dependencies

Create a virtual env and install packages:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**`requirements.txt`**

```text
pandas
python-dotenv
openai==1.*
chromadb
pyarrow
openpyxl
tiktoken
```

> `pyarrow` for Parquet, `openpyxl` for Excel, `tiktoken` if you want to estimate tokens.

### 2) Azure OpenAI environment variables

Create `.env` in the repo root:

```ini
AzureOpenAI_ENDPOINT=https://<your-openai-resource>.openai.azure.com/
AzureOpenAI_API_KEY=<your-azure-openai-key>
AzureOpenAI_API_VERSION=2024-05-01-preview

# Use your deployment names (not model families)
AzureOpenAI_CHAT_MODEL=gpt-4o-mini
AzureOpenAI_EMBEDDING_MODEL=text-embedding-3-small
```

> Ensure you **deploy** both a **chat** model and an **embedding** model in your Azure OpenAI resource, and use those **deployment names** here.

***

## Data format

Each file should have columns like:

*   `id` (string or numeric; auto‑generated if missing)
*   `date` (e.g., `2025-06-01`)
*   `merchant`
*   `category`
*   `amount` (numeric)
*   `notes` (free text)

During ingest, the app:

*   Reads all files from the folder you specify (non‑recursive)
*   Ensures `id` is unique per file (prefixes with filename stem)
*   Builds row‑level text (one “document” per row) for embeddings
*   Persists a **normalized** combined table → `store/expenses.parquet`

***

## How it works (end‑to‑end)

    [Expense files (CSV/XLSX)]     ──►  Ingest  ──►  [Chroma vectors]
              │                                   │
              └────────────────────────►  [Parquet table for analytics]
     
    Ask("question") ──► Plan with LLM ───────────────► If intent=analytics:
                                                  run plan with pandas on ALL rows
                                                  (correct totals & groups)
                                       └── else: RAG (Chroma) + LLM explanation

### Query planner (LLM → JSON plan)

The system prompt instructs the LLM to output **only JSON** with keys like:

*   `intent`: `"analytics"` or `"lookup"`
*   `filters`: list of `{column, op, value}`
*   `group_by`: list of column names
*   `aggregations`: list of `{op, column, alias}` where `op ∈ {sum, avg, min, max, count}`
*   `order_by`: list of `{column, direction}`
*   `select`, `limit`
*   `date_parts`: list of `year|month|day` to expose those fields

**Example plan** (analytics):

```json
{
  "intent": "analytics",
  "filters": [{"column": "year", "op": "eq", "value": 2025}],
  "group_by": ["category"],
  "aggregations": [{"op": "sum", "column": "amount", "alias": "total_amount"}],
  "order_by": [{"column": "total_amount", "direction": "desc"}],
  "limit": 10,
  "date_parts": ["year"]
}
```

The **executor** applies this plan with pandas: filters → date parts → groupby/aggregations → order/limit.  
For **lookup** intent, the app does **semantic search** in Chroma and lets the LLM answer from retrieved rows.

***

## Running the app

```bash
python app.py
```

Interactive menu:

1.  **Load and vectorize expense data**
    *   Enter a folder path (e.g., `data/`).
    *   The app reads all CSV/XLSX/XLS files, creates embeddings, writes to Chroma, and saves `store/expenses.parquet`.

2.  **Interactive Chat**
    *   Type questions naturally. Examples:
        *   “Summarize my expenses by category for 2025”
        *   “Total Uber spend between 2025-06-01 and 2025-06-30”
        *   “Show similar expenses to my ‘airport to hotel’ ride”
        *   “What was my biggest travel expense in July 2025?”
    *   The agent will:
        *   For analytics: compute a **deterministic** table and (optionally) ask the LLM to **explain** it.
        *   For lookup: retrieve top matches and answer from context.

3.  **Exit**

***

## Design choices & guidance

*   **Embedding granularity:** **Per row**.  
    Each transaction is embedded independently to preserve semantics and enable precise retrieval.

*   **`k` for retrieval:**  
    Used only for **lookup** questions. Start with `k=12` (or 8–25). For analytics, we compute on **all rows**, so `k` is irrelevant there.

*   **Chunking:**  
    Not needed for typical expense rows (short). Consider it only for very long texts (e.g., OCR’d receipts/PDFs). The wrong totals you saw earlier were **not** a chunking issue; they were a **retrieval‑for‑aggregation** issue.

*   **Early return vs `else` in `ask()`**:  
    The analytics path ends with `return`, so the lookup path can live after it (cleaner indentation) and still be mutually exclusive.

***

## Troubleshooting

*   **Numbers seem wrong for totals/breakdowns**  
    Ensure you ran **ingest** after adding files. Analytics uses `store/expenses.parquet`.  
    Verify the planner returned `intent="analytics"` (you can temporarily print the plan).

*   **`fillna` no‑op**  
    In `read_expenses`, use `df = df.fillna("")` (assign back), not just `df.fillna("")`.

*   **Model/Deployment errors**
    *   Make sure **AzureOpenAI\_CHAT\_MODEL** and **AzureOpenAI\_EMBEDDING\_MODEL** are your **deployment names**.
    *   Verify `AzureOpenAI_API_VERSION` is supported by your resource.

*   **Excel date parsing**  
    The app coerces dates via `pd.to_datetime(..., errors="coerce")`. Invalid dates become `NaT`.

*   **Start fresh**  
    Delete `./chroma_db` and `./store/expenses.parquet` to rebuild.

***

## Security & data handling

*   Keep your `.env` out of source control.
*   Data stays local by default (Chroma on disk, Parquet on disk).
*   Azure OpenAI handles embeddings & chat requests within your Azure subscription.

***

## Roadmap ideas (optional)

*   Add more filter ops (`in`, `startswith`, `endswith`) and date shortcuts (“last 30 days”).
*   Support metadata filters in vector search (if switching to a cloud vector DB).
*   Add unit tests for the executor and a schema validator for plans.

***

## License

Choose a license that fits your use (e.g., MIT). Example:

    MIT License — see LICENSE for details.

***

## Acknowledgments

*   Azure OpenAI (embeddings + chat)
*   Chroma (local vector database)
*   Pandas (analytics)

***

### Appendix: Sample Q→Plan→Result flow

**Q:** “Break down my 2025 expenses by category, highest first (top 10).”  
**Plan (LLM):** see example JSON above.  
**Executor result:** a deterministic table with `category` and `total_amount`.  
**Explanation:** optional 3–5 sentence summary generated by the LLM from the computed table.

***

That’s it—commit this `README.md`, and your repo will be self‑explanatory for anyone cloning it (including future you).
