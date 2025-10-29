# ragbot
AI-Powered RAG Chatbot using LangChain, FAISS &amp; Streamlit | Smart Q&amp;A from CSV
 SDG Insight Chatbot (RAG-powered)
A Retrieval-Augmented Generation (RAG) chatbot that provides intelligent answers based on Sustainable Development Goals (SDG) indicators. Built with LangChain, FAISS, Streamlit, and HuggingFace models.

📚 Project Overview
This chatbot helps users explore SDG data by answering natural language questions.
It uses:

Document loading: CSV data → cleaned & chunked
Embedding: sentence-transformers/all-MiniLM-L6-v2
Vector store: FAISS for fast semantic retrieval
LLM: google/flan-t5-base via HuggingFaceHub
Frontend: Streamlit-based chatbot UI
📁 Project Structure
rag-chatbot/
│
├── data/                       # SDG dataset (CSV)
│   └── sdg.csv
│
├── vectorstore/               # FAISS index
│   └── faiss_index/
│
├── chatbot/                   # Core modules
│   ├── __init__.py
│   ├── data_loader.py         # CSV to documents
│   ├── rag_pipeline.py        # Vector store builder
│   ├── generate_sample_qa.py  # Script to export sample Q&A
│   └── app.py                 # Streamlit chatbot UI
│
├── app/                       # Streamlit
│   └── app.py                 # Streamlit chatbot UI
│
├── sample_outputs/            # Sample generated Q&A
│   └── sample_qa.csv
│
├── .env                       # API keys
├── requirements.txt           # Libraries
└── README.md                  # Project overview
🛠️ Tech Stack
Component	Technology
LLM	google/flan-t5-base via HuggingFaceHub
Embeddings	sentence-transformers/all-MiniLM-L6-v2
Vector Store	FAISS
Frameworks	LangChain
Dataset Source	World Bank SDG Indicators
Dashboard	Streamlit
📦 Installation
1. Clone the Repository
git clone https://github.com:yashdew3/ragbot.git
cd rag-chatbot
2. Create Virtual Environment
python -m venv venv
source venv/bin/activate     # on Windows: venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Set Environment Variables
Create a .env file in the root:

HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
🔑 Get a free API key from https://huggingface.co/settings/tokens

🧪 Sample Q&A Export (Optional)
If you want to generate sample questions/answers:

# Use generate_sample_qa.py (optional)
python chatbot/generate_sample_qa.py
🧠 How It Works
Data Loading – Loads & previews sdg.csv

Document Conversion – Chunks data into LangChain-compatible documents

Embedding + FAISS – Converts docs into embeddings, stores in FAISS

QA Chain – Uses flan-t5-base + retrieved chunks to generate answers

Chat UI – Streamlit frontend for question answering

💬 Sample Questions
Question	Answer
What is the access to electricity in rural areas in Africa?	68.98945175
How has access to clean fuels changed since 2000?	East Asia & Pacific (excluding low income)
Which countries lead in renewable energy use?	East Asia & Pacific
📂 View full list in: sample_outputs/sample_qa.csv
✅ Run the Chatbot
streamlit run chatbot/app.py
📩 Requirements
Python 3.8+
Libraries listed in requirements.txt
💡 Use Cases
Data analytics dashboards

Automated report Q&A bots

Domain-specific chat assistants (healthcare, finance, education)

Research exploration tools

Internal tools for enterprise CSV datasets

🔐 Privacy & Offline Use
This chatbot supports offline embeddings using Hugging Face or Llama models. You are free from vendor lock-in and API rate limits.

