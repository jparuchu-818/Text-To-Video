# Invoice QA Chatbot 💬📄

An AI-powered **Question Answering system for Invoices** built using **LangChain, FastAPI, Streamlit, and ChromaDB**.  
This project demonstrates how Retrieval-Augmented Generation (RAG) can be applied to extract and answer questions from financial documents.

---

## 🚀 Features
- 📂 **Invoice Upload** – Upload PDF or image-based invoices.  
- 🔍 **Vector Search** – Store embeddings in **ChromaDB** for fast retrieval.  
- 🤖 **LLM-Powered Q&A** – Local inference using **Ollama + LLaMA models**.  
- 🌐 **REST API** – FastAPI backend for integration.  
- 🎨 **Interactive UI** – Streamlit-based user interface.  
- 📊 **Evaluation** – Stubbed modules for retrieval precision & RAGAS scoring.  
- 🧪 **Unit Tests** – Basic test suite for core functions.  

---

## 📂 Project Structure
invoice-qa-chatbot/
│
├── app/
│ ├── functions.py # Core business logic
│ ├── text.py # Preprocessing utilities
│ ├── retriever.py # Vector DB wrapper
│ ├── evaluator.py # Fake evaluation metrics
│ ├── config.py # Config & environment settings
│ ├── database.py # Chroma/FAISS mock database
│
├── api/
│ ├── main.py # FastAPI entrypoint
│ ├── routes.py # Upload & query routes
│ ├── schemas.py # Pydantic models
│
├── ui/
│ ├── streamlit_ui.py # Streamlit web app
│ ├── dashboard.py # Analytics dashboard
│
├── tests/ # Pytest test cases
│
├── scripts/ # Utility scripts
│
├── data/ # Sample data/invoices
│
├── requirements.txt
├── README.md
├── LICENSE
