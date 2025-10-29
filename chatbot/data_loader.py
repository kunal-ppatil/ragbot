"""
Module: data_loader.py
Description: Loads and processes the SDG dataset to prepare it for RAG-based chatbot use.
"""

import pandas as pd
from langchain.docstore.document import Document
import os

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the SDG CSV dataset into a DataFrame.
    
    Args:
        csv_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df

def analyze_data(df: pd.DataFrame) -> None:
    """
    Display basic analysis and information about the dataset.

    Args:
        df (pd.DataFrame): The dataset DataFrame.
    """
    print("---- Dataset Preview ----")
    print(df.head(), "\n")

    print("---- Dataset Info ----")
    print(df.info(), "\n")

    print("---- Null Values ----")
    print(df.isnull().sum(), "\n")

def convert_to_documents(df: pd.DataFrame) -> list:
    """
    Convert each row of the dataset into a LangChain Document.

    Args:
        df (pd.DataFrame): Cleaned SDG dataset.

    Returns:
        List[Document]: List of LangChain Document objects.
    """
    documents = []
    
    for _, row in df.iterrows():
        content = (
            f"Goal: {row.get('Goal')}\n"
            f"Target: {row.get('Target')}\n"
            f"Indicator: {row.get('Indicator')}\n"
            f"Description: {row.get('Description')}\n"
        )
        documents.append(Document(page_content=content))
    
    print(f"Converted {len(documents)} rows to Document format.")
    return documents

if __name__ == "__main__":
    # Test loading and conversion
    dataset_path = os.path.join("data", "sdg.csv")
    df = load_data(dataset_path)
    analyze_data(df)
    docs = convert_to_documents(df)
