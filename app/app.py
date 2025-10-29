# ui/app.py

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="🌍 SDG Chatbot", layout="wide")
st.title("🌍 Sustainable Development Goals (SDG) Chatbot")
st.markdown("Ask anything about SDGs, indicators, or global development 📊")

# Load vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# Load local LLM with HuggingFacePipeline
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

# Initialize QA chain
@st.cache_resource
def initialize_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = load_llm()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

qa_chain = initialize_qa_chain()

# Chat input
query = st.text_input("Enter your question here 👇")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(query)
        st.success("Here's the answer:")
        st.write(result)

# Footer
st.markdown("---")
st.markdown("🛠️ Built using LangChain, FAISS, HuggingFace Transformers, and Streamlit")
