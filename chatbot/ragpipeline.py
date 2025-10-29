import os
from dotenv import load_dotenv
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables (HuggingFace token if needed)
load_dotenv()

DATA_PATH = "data/sdg.csv"
VECTOR_STORE_PATH = "vectorstore/faiss_index"

# --------------- STEP 1: Load & Analyze CSV ---------------- #
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    print("---- Dataset Preview ----")
    print(df.head(), "\n")
    print("---- Dataset Info ----")
    print(df.info(), "\n")
    print("---- Null Values ----")
    print(df.isnull().sum(), "\n")
    return df

# --------------- STEP 2: Convert to Documents -------------- #
def convert_to_documents(df):
    documents = []
    for i, row in df.iterrows():
        metadata = {
            "country": row["Country Name"],
            "code": row["Country Code"],
            "indicator": row["Indicator Name"],
            "indicator_code": row["Indicator Code"]
        }

        for year in range(1990, 2019):
            year_str = str(year)
            if pd.notnull(row.get(year_str)):
                content = f"{row['Country Name']} - {row['Indicator Name']} in {year_str}: {row[year_str]}"
                documents.append(Document(page_content=content, metadata=metadata))
    
    print(f"Converted {len(documents)} rows to Document format.")
    return documents

# --------------- STEP 3: Build FAISS Vector Store ---------- #
def build_vector_store(documents):
    print("Creating new FAISS index...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(documents, embeddings)

    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"Saved FAISS index to: {VECTOR_STORE_PATH}")
    return vectorstore

# --------------- STEP 4: Load Vector Store ----------------- #
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# --------------- MAIN EXECUTION ---------------------------- #
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    documents = convert_to_documents(df)

    if not os.path.exists(VECTOR_STORE_PATH + ".index"):
        vectorstore = build_vector_store(documents)
    else:
        print("Loading existing FAISS index...")
        vectorstore = load_vector_store()

    # Optional: Run a sample query
    query = "What is the electricity access in Arab World in 2015?"
    docs = vectorstore.similarity_search(query, k=3)

    print("\nTop 3 relevant results:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")
