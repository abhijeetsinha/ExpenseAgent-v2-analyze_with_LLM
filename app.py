import os
import sys
import pandas as pd
import uuid
import json
import re
from dotenv import load_dotenv
from typing import List, Dict, Iterable
from pathlib import Path

# Azure OpenAI imports
from openai import AzureOpenAI

# Chroma
import chromadb
from chromadb.config import Settings



# Load environment variables from .env file
load_dotenv()

AzureOpenAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AzureOpenAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AzureOpenAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AzureOpenAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL")
AzureOpenAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL") 

print("AZURE_OPENAI_ENDPOINT:", AzureOpenAI_ENDPOINT)
print("AZURE_OPENAI_API_KEY:", AzureOpenAI_API_KEY)
print("AZURE_OPENAI_API_VERSION:", AzureOpenAI_API_VERSION)
print("AZURE_OPENAI_CHAT_MODEL:", AzureOpenAI_CHAT_MODEL)
print("AZURE_OPENAI_EMBEDDING_MODEL:", AzureOpenAI_EMBEDDING_MODEL)

if not all([AzureOpenAI_ENDPOINT, AzureOpenAI_API_KEY, AzureOpenAI_API_VERSION, AzureOpenAI_CHAT_MODEL, AzureOpenAI_EMBEDDING_MODEL]):
    print("One or more environment variables are missing. Please check your .env file.")
    sys.exit(1)


# System prompts for planner. It should generate a JSON plan only.
PLAN_SYS = (
    "You translate a natural-language question about an expenses table into a JSON plan. "
    "Schema: id:string, date:date, merchant:string, category:string, amount:number, notes:string. "
    "Only output valid JSON with keys: intent ('analytics'|'lookup'), filters (list), "
    "group_by (list), aggregations (list of {op,column,alias}), order_by (list of {column,direction}), "
    "select (list), limit (int or null), date_parts (list of 'year'|'month'|'day'). "
    "If the user clearly wants totals/averages or grouping, set intent='analytics'. "
    "If they want examples/similar rows/freeform text answers, set intent='lookup'."
)

ALLOWED_AGGS = {"sum", "avg", "min", "max", "count"}



# Initialize Azure OpenAI client
azureopenai_client = AzureOpenAI(
    api_key = AzureOpenAI_API_KEY,
    api_version = AzureOpenAI_API_VERSION,
    azure_endpoint = AzureOpenAI_ENDPOINT
)



# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path = "./chroma_db", settings = Settings(allow_reset = False))
COLLECTION_NAME = "expenses_collection"


# Get or create ChromaDB collection
def get_or_create_collection():
    try:
        return chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        return chroma_client.create_collection(COLLECTION_NAME)
    


# Utilities
def read_expenses(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    print(f"Reading file: {file_path} with extension: {ext}")
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xls", ".xlsx"):
        df =  pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

    df = df.fillna("")
    if "id" not in df.columns:
        df.insert(0, "id", [str(uuid.uuid4()) for _ in range(len(df))])
    
    df["id"] = df ["id"].astype(str)
    return df



def row_to_text (row: pd.Series) -> str:
    parts = []    
    for col in row.index:
        if col == "id":
            continue
        val = str(row[col]).strip()
        if val != "":
            parts.append(f"{col}: {val}")
    return "\n".join(parts)



def embed_text(texts : List[str]) -> List[List[float]]:
    response = azureopenai_client.embeddings.create(model=AzureOpenAI_EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in response.data]



# ingest data
def ingest_expenses(folder: str):
    folder_path = Path(folder)
    print(f"Ingesting expenses from folder: {folder_path}")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder path {folder_path} does not exists!")
    
    files = [file for file in folder_path.iterdir() if file.is_file()]
    print(f"Found {len(files)} files in folder: {folder_path}")
    print(f"Files: {files}")
    ingest_files(files)
    
    if not files:
        raise FileNotFoundError(f"No files found in folder {folder_path}")
    


def ingest_files(files: Iterable[Path]):
    all_docs: List[str] = []
    all_metadatas: List[Dict] = []
    all_ids: List[str] = []
    total_rows = 0

    for f in files:
        df= read_expenses(f)
        rows = len(df)
        print(f"Dataframe shape: {df.shape}")

        
        # Ensure IDs are globally unique across multiple files by prefixing filename
        # (If your 'id' column is truly globally unique already, you can skip this.)
        df["id"] = df.apply(lambda x: f"{f.stem}:{x['id']}", axis=1)

        for _, row in df.iterrows():
            all_ids.append(str(row["id"]))
            all_docs.append(row_to_text(row))  # what we embed & store for retrieval
            # Keep original fields as metadata for display/filters later (optional)
            meta = {k: (str(v) if pd.notna(v) else "") for k, v in row.to_dict().items()}
            all_metadatas.append(meta)

        total_rows += rows
        print(f"Processed {rows} rows from file {f.name}")
    print(f"Total rows processed from all files: {total_rows}")
    #print("docs:", all_docs)
    #print("ids:", all_ids)
    #print("metadatas:", all_metadatas)

    batch_size = 128
    vectors: List[List[float]] = []
    for i in range(0, len(all_docs), batch_size):
        batch_texts = all_docs[i:i + batch_size]
        batch_vectors = embed_text(batch_texts)
        vectors.extend(batch_vectors)
        #print(f"Vectors: {vectors}")
        print(f"Processed batch {i // batch_size + 1}, total vectors: {len(vectors)}")
    print(f"Generated {len(vectors)} embeddings.")

    collections = get_or_create_collection()
    collections.add(
        documents = all_docs,
        metadatas = all_metadatas,
        ids = all_ids,
        embeddings = vectors
    )
    print(f"Inserted {len(all_docs)} records into ChromaDB collection '{COLLECTION_NAME}'.")

    df_all = pd.DataFrame(all_metadatas)
    df_all["date"] = pd.to_datetime(df_all.get("date", ""), errors='coerce')
    df_all["amount"] = pd.to_numeric(df_all.get("amount", ""), errors='coerce')
    df_all["year"] = df_all["date"].dt.year
    Path("store").mkdir(parents=True, exist_ok= True)
    df_all.to_parquet("store/expenses.parquet", index=False)
    print("Saved all metadata to store/expenses.parquet")

# Ask questions

def retrieve(query: str, k: int = 20):
    q_vec = embed_text([query])[0]
    #print("Query vector:", q_vec)

    col = get_or_create_collection()
    #print("Collection:", col)

    res = col.query(query_embeddings=[q_vec], n_results=k)  # similarity search
    #print("Similarity search result (this is a list of lists):", res)
    # res contains "documents", "metadatas", "ids", "distances"


    # since res is a list of lists, we are iterating over 1 value of each list and 
    # creating a simple list of dicts. Nothing fancy here.
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        })

    print(f"Retrieved {len(hits)} hits for query: '{query}'")
    #print("Hits:", hits)
    return hits





def answer_with_context(question: str, hits: List[Dict]) -> str:
    # Build a tiny prompt with the retrieved rows
    snippets = []
    for h in hits:
        snippets.append(f"- ID {h['id']}\n{h['document']}")
    context = "\n\n".join(snippets) if snippets else "No matching expenses found."
    print("\n=== Context ===")
    #print(context)
    print("=== End Context ===\n")

    system = (
        "You are a helpful assistant that answers questions about the user's expenses. "
        "Use ONLY the provided context to answer. If the answer isn't in the context, say you don't know."
    )

    print("\n=== System ===")
    print(system)
    print("=== End System ===\n")

    user = f"Question: {question}\n\nContext:\n{context}"
    print("\n=== Prompt ===")
    print(user)
    print("=== End Prompt ===\n")   



    resp = azureopenai_client.chat.completions.create(
        model=AzureOpenAI_CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.0,
    )
    return resp.choices[0].message.content



def plan_query_with_LLM(question:str) -> Dict:
    resp = azureopenai_client.chat.completions.create(
        model=AzureOpenAI_CHAT_MODEL,
        messages=[{"role": "system", "content": PLAN_SYS},
                  {"role": "user", "content": f"Question: {question}\nReturn only JSON"}],
        temperature=0.0,
    )
    txt = resp.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except Exception as e:
        return {"intent": "lookup"}



def load_table() -> pd.DataFrame:
    path = Path ("store/expenses.parquet")
    if not path.exists():
        raise FileNotFoundError("No expenses data found. Please ingest data first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df.get("date", ""), errors='coerce')
    df["amount"] = pd.to_numeric(df.get("amount", ""), errors='coerce')
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    return df



def apply_filters(df: pd.DataFrame, filters: List) -> pd.DataFrame:
    output_df=df.copy()
    for f in filters or []:
        col = f.get("column")
        op = f.get("op")
        value = f.get("value")
        if col not in df.columns:
            continue
        if op == "eq": output_df = output_df[output_df[col] == value] #boolean indexing
        elif op == "neq": output_df = output_df[output_df[col] != value]
        elif op == "contains": output_df = output_df[output_df[col].astype(str).str.contains(str(value))]
        elif op == "gte": output_df = output_df[output_df[col] >= value]
        elif op == "lte": output_df = output_df[output_df[col] <= value]
    return output_df


def execute_plan (df: pd.DataFrame, plan: Dict) -> pd.DataFrame:
    df = df.copy()
    df = apply_filters(df, plan.get("filters"))

    # select date parts from JSON plan, if any.
    for part in plan.get("date_parts") or []:
        if part=="year":
            df["year"] = df["date"].dt.year
        if part=="month":
            df["month"] = df["date"].dt.month
        if part=="day":
            df["day"] = df["date"].dt.day

    # Groupings and aggregations
    group_by = plan.get("group_by") or []
    aggregates = plan.get("aggregations") or []
    if aggregates or group_by:
        agg_dict = {}
        for agg in aggregates:
            op = agg.get("op")
            col = agg.get("column")
            alias = agg.get("alias") or f"{op}_{col}"
            if op not in ALLOWED_AGGS or col not in df.columns:
                print(f"Skipping invalid aggregation: {agg}")
                continue
            if op == "avg":
                agg_dict[alias] = (col, "mean")  # pandas uses 'mean' instead of 'avg'
            elif op == "count":
                agg_dict[alias] = (col, "count")
            else:
                agg_dict[alias] = (col, op)
        
        if agg_dict:
            result = df.groupby(group_by).agg(**agg_dict).reset_index()
        else:
            result = df[group_by].drop_duplicates()
    else:
        #select
        select = plan.get("select") or []
        df = df[select] if select else df

    #order by and limits
    for order_by in plan.get("order_by") or []:
        col = order_by.get("column")
        direction = (order_by.get("direction") or "desc").lower()
        if col in result.columns:
            result = result.sort_values(by=col, ascending=(direction == "asc"))
    if plan.get("limit"):
        result = result.head(int(plan["limit"]))
    return result



def ask(question: str):
    plan = plan_query_with_LLM(question)
    print("\n=== Plan ===")
    print(plan)
    print("=== End Plan ===\n")

    if plan.get("intent") == "analytics":
        df = load_table()
        print("Analytics query.")
        result = execute_plan(df,plan)
        #print top 20 rows
        print("\n========= Computed result (first 20 rows) ===========")
        print(result.head(20).to_string())
        return


    # Fallback to semantic lookup for non-analytics queries
    hits = retrieve(question)
    ans = answer_with_context(question, hits)
    print("\n=== Answer ===")
    print(ans)
    print("\n=== Top matches (debug) ===")
    for h in hits:
        md = h["metadata"]
        basic = f"{md.get('date','?')} • {md.get('merchant','?')} • {md.get('category','?')} • {md.get('amount','?')}"
        print(f"{h['id']}: {basic}  (distance={h['distance']:.4f})")


# main chat function
def main():

    """Main application entry point"""
    print("💰 Welcome to the Expense Analysis Agent! 💰")
    print("Powered by Azure OpenAI")

    while True:
        print("="*50)
        print("1. Load and vectorize expense data")
        print("2. Interactive Chat")
        print("3. Exit")
        print(""*50)

        user_input = input("Select an option (1-3): ").strip()
        if user_input == "1":
            folder_path = input("Enter folder path (or press Enter for default 'data/sample_expenses'): ").strip()
            if not folder_path:
                folder_path = "data"
            ingest_expenses(folder_path)
        
        elif user_input == "2":
            print("Enter your question about expenses: ")
            question = input().strip()
            ask(question)

        elif user_input == "3":
            print("👋 Thank you for using the Expense Analysis Agent!")
            break

        else:
            print("❌ Invalid option. Please select 1-3.")


if __name__ == "__main__":
    main()
