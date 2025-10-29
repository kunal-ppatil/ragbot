import os
import csv
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Load FAISS vector store
index_path = "vectorstore/faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

# Use HuggingFaceEndpoint instead of deprecated HuggingFaceHub
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    temperature=0.5,
    max_length=256
)

# Define custom prompt (optional)
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

# Sample questions
sample_questions = [
    "What is the access to electricity in rural areas in Africa?",
    "How is gender equality progressing globally?",
    "What are the trends in global poverty rates?",
    "How does climate change affect small island nations?",
    "What measures are being taken to promote sustainable cities?"
]

# Generate answers and save to CSV
def generate_sample_qa(output_csv="sample_qa.csv"):
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Answer"])
        for question in sample_questions:
            print(f"Fetching answer for: {question}")
            try:
                response = qa_chain.invoke({"query": question})
                answer = response.get("result", "No answer found")
            except Exception as e:
                answer = f"Error: {e}"
            writer.writerow([question, answer])
            print(f"Answer: {answer}\n")

if __name__ == "__main__":
    generate_sample_qa()
