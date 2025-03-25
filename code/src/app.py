import subprocess
import sys

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Install the packages via subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "openpyxl", "sentence-transformers", "faiss-cpu"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])

import pandas as pd

import pkg_resources
print(pkg_resources.get_distribution("sentence-transformers"))
import faiss
import gradio as gr

from sentence_transformers import SentenceTransformer
import torch

# Load Sentence Transformer Model (Enable Parallel Encoding if GPU available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
#model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device=device)

# FAISS Index Setup
embedding_dim = 384  # all-MiniLM-L6-v2 outputs 384-dimensional vectors
index = faiss.IndexFlatL2(embedding_dim)

# CSV File Path
csv_path = "customer_support_tickets.csv"
#csv_path = "data-sets/customer_support_tickets_200K.csv"
#csv_path = "data-sets/customer_support_tickets_800K.csv"
chunksize = 5000  # Increase chunk size for better performance

# Track total rows processed
total_rows = 0

# Process CSV in chunks
for chunk_id, df in enumerate(pd.read_csv(csv_path, encoding="utf-8", low_memory=False, chunksize=chunksize)):
    try:
        # Convert all fields to string and fill NaN with empty string
        df = df.astype(str).fillna("")

        # Faster concatenation using .agg()
        df["text"] = df[["Ticket ID", "Customer Name", "Customer Email", "Ticket Type",
                               "Ticket Subject", "Ticket Description", "Ticket Status",
                               "Resolution", "Ticket Priority", "Created Date"]].agg(" ".join, axis=1)

        # Encode text in batches with multi-threading
        with torch.cuda.amp.autocast():
            embeddings = model.encode(df["text"].values(), convert_to_numpy=True,
                                  batch_size=1024, show_progress_bar=True).astype("float32")

        # Add embeddings to FAISS index
        index.add(embeddings)
        total_rows += len(df)

        print(f"✅ Processed Chunk {chunk_id + 1} - Total Rows: {total_rows}")

    except Exception as e:
        print(f"❌ Error in Chunk {chunk_id + 1}: {e}")

# Save FAISS index to disk
faiss.write_index(index, "faiss_index.idx")
print("🎉 FAISS index built and saved successfully!")
print(f"📌 Total Records Indexed: {index.ntotal}")

subprocess.check_call([sys.executable, "-m", "pip", "install", "rapidfuzz", "pandas", "faiss-cpu"])
#!pip install rapidfuzz pandas faiss-cpu
from rapidfuzz import fuzz
import pandas as pd
import re



def search_data(query, filter_value, top_k=5, max_distance=0.9, min_similarity=80):
    filter_value = filter_value.lower().strip()  # Normalize input

    # ✅ Step 3: Exact Match Search
    exact_match = df[df["text"].str.contains(re.escape(filter_value), case=False, na=False, regex=True)]

    if not exact_match.empty:
        print("\n✅ Exact Match Found!")
        return format_results(exact_match.iloc[:top_k])

    print("\n⚠️ No Exact Match Found. Performing FAISS Search...")

    # ✅ Step 4: FAISS Search (If Embeddings Exist)
    query_embedding = model.encode(query).astype("float32").reshape(1, -1)
    D, I = index.search(query_embedding, top_k)

    filtered_results = []

    for i in range(len(I[0])):
        idx = I[0][i]
        if idx == -1 or idx >= len(df):
            continue

        text_result = df.iloc[idx]["text"]

        if D[0][i] <= max_distance and fuzz.partial_ratio(filter_value, text_result.lower()) >= min_similarity:
            filtered_results.append(df.iloc[idx])

        if len(filtered_results) >= top_k:
            break

    # ✅ Step 5: Handle No Results
    if not filtered_results:
        substring_match = df[df["text"].str.contains(re.escape(filter_value), case=False, na=False)]
        if not substring_match.empty:
            return format_results(substring_match.iloc[:top_k])

    return format_results(pd.DataFrame(filtered_results).iloc[:top_k]) if filtered_results else [{"message": f"No relevant results for '{filter_value}' found."}]

def format_results(df_results):
    """Formats search results into a column-value dictionary."""
    formatted = []
    for _, row in df_results.iterrows():
        formatted.append({col: row[col] for col in df_results.columns})
    return formatted

def chat_interface(query, filter_value):
    print(f"🔍 Received Query: {query}, Filter: {filter_value}")  # Debugging print
    if not query or not filter_value:
        return "⚠️ Error: Please provide both a query and a filter value."

    try:
        results = search_data(query, filter_value,10)  # ✅ Ensuring both arguments are passed correctly
    except Exception as e:
        return f"⚠️ Error while searching: {str(e)}"

    if not results:
        return "❌ No relevant results found."

    # ✅ Format results properly
    if isinstance(results, list) and all(isinstance(res, dict) for res in results):
        formatted_results = []
        for res in results:
            formatted_text = "\n".join([f"**{key}**: {value}" for key, value in res.items()])
            formatted_results.append(formatted_text)
        return "\n\n---\n\n".join(formatted_results)
    else:
        return "\n".join(results)  # If results are strings (error messages)

# ✅ Gradio UI
gr.Interface(
    fn=chat_interface,
    inputs=[
        gr.Textbox(label="Enter your query"),
        gr.Textbox(label="Enter filter value (e.g., Policy Number)")
    ],
    outputs="text",
    title="Chatbot with Dynamic Search"
).launch(share=True)