from langchain.vectorstores import Chroma

class InvoiceRetriever:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.db = Chroma(collection_name="invoices", embedding_function=embedding_model)

    def add_document(self, doc):
        print(f"[Retriever] Added document: {doc[:50]}...")

    def query(self, question: str, top_k: int = 3):
        print(f"[Retriever] Query: {question}, top_k={top_k}")
        return [{"answer": "Fake invoice answer", "score": 0.88}]