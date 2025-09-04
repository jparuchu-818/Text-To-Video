import os
import re
import shutil
from typing import List, Tuple, Dict

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
from pydantic import Field

from langchain_huggingface import HuggingFaceEmbeddings

# Directory to store the Chroma vector index
CHROMA_DIR = "./chroma_index"

# Pattern to identify invoice headers in PDF files
HEADER_RE = re.compile(r"INVOICE\s*#\s*\d+", re.I)

# Define the embedding model (used to convert text into vector format)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

def split_invoice_pages_by_header(pages: List[Document]) -> List[Document]:
    """
    Splits a list of PDF pages into separate documents based on invoice headers.
    Each invoice starts with a page that matches the HEADER_RE pattern.
    """
    grouped_docs: List[Tuple[str, List[str]]] = []

    for page in pages:
        header_match = HEADER_RE.search(page.page_content)
        if header_match:
            grouped_docs.append((header_match.group(0), [page.page_content]))
        elif grouped_docs:
            grouped_docs[-1][1].append(page.page_content)
        else:
            grouped_docs.append(("page", [page.page_content]))

    return [Document(page_content="\n".join(pages), metadata={"header": header}) for header, pages in grouped_docs]

def create_vector_index(file_paths: List[str]) -> Tuple[Chroma, int]:
    """
    Loads documents from file paths (PDF, CSV, images),
    splits them into smaller chunks, and creates a vector index using Chroma.

    Returns:
        - A Chroma index built from the documents.
        - The number of raw documents before chunking.
    """
    raw_documents: List[Document] = []

    for path in file_paths:
        lower_path = path.lower()

        if lower_path.endswith(".pdf"):
            pdf_pages = PyPDFLoader(path).load()
            raw_documents.extend(split_invoice_pages_by_header(pdf_pages))

        elif lower_path.endswith((".csv", ".tsv")):
            df = pd.read_csv(path, sep="\t" if lower_path.endswith(".tsv") else ",")
            for _, row in df.iterrows():
                content = "\n".join(f"{k}: {v}" for k, v in row.items())
                metadata = {k.lower(): str(v).lower() for k, v in row.items()}
                raw_documents.append(Document(page_content=content, metadata=metadata))

        elif lower_path.endswith((".png", ".jpg", ".jpeg")):
            raw_documents.extend(UnstructuredImageLoader(path).load())

    document_count = len(raw_documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunked_documents = splitter.split_documents(raw_documents)

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    index = Chroma.from_documents(chunked_documents, embedding_model, persist_directory=CHROMA_DIR)
    return index, document_count

def build_cross_encoder_retriever(index: Chroma) -> BaseRetriever:
    """
    Builds a hybrid retriever for better document search results.

    This retriever combines:
    1. MMR (vector) search — finds documents that are semantically similar.
    2. Cross-encoder re-ranking — re-orders results using deeper context.
    3. Keyword fallback — adds documents containing exact word matches from the query.

    Returns:
        A custom retriever that ranks documents more accurately.
    """
    # Step 1: Build a basic retriever using Max Marginal Relevance (MMR)
    vector_retriever = index.as_retriever(search_type="mmr", search_kwargs={"k": 60, "lambda_mult": 0.5})

    # Step 2: Load cross-encoder model (used for smarter re-ranking)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Weight factor: how much to trust vector similarity vs cross-encoder (0.3 = 30% vector, 70% encoder)
    alpha = 0.3

    # Step 3: Build an inverted index for fallback exact-match boosting
    collection = index._collection.get(include=["documents", "metadatas"])
    inverted_index: Dict[str, List[int]] = {}

    for idx, (doc_text, metadata) in enumerate(zip(collection["documents"], collection["metadatas"])):
        combined_text = (doc_text or "").lower() + " " + " ".join(str(v).lower() for v in metadata.values())
        for token in re.findall(r"[\w@.\-]{3,}", combined_text):
            inverted_index.setdefault(token, []).append(idx)

    class SimpleFusionRetriever(BaseRetriever):
        """
        Custom retriever class that:
        - Starts with vector-based matches.
        - Adds keyword-matched docs.
        - Removes duplicates.
        - Re-ranks using cross-encoder.
        """
        base_retriever: BaseRetriever = Field(...)
        cross_encoder: CrossEncoder = Field(...)
        alpha: float = Field(...)
        inv_index: Dict[str, List[int]] = Field(...)
        store: Chroma = Field(...)

        @staticmethod
        def _tokenize(text: str) -> List[str]:
            return re.findall(r"[\w@.\-]{3,}", text.lower())

        def get_relevant_documents(self, query: str):
            # Get documents using vector similarity
            vector_hits = self.base_retriever.vectorstore.similarity_search_with_relevance_scores(query, k=60)
            docs, vector_scores = (list(t) for t in zip(*vector_hits)) if vector_hits else ([], [])

            # Add exact token matches from inverted index
            matching_indices = set()
            for token in self._tokenize(query):
                matching_indices.update(self.inv_index.get(token, []))

            for idx in matching_indices:
                docs.append(Document(
                    page_content=collection["documents"][idx],
                    metadata=collection["metadatas"][idx]
                ))
                vector_scores.append(1.0)  # Give full score to exact match

            # Deduplicate by content snippet (first 100 chars)
            seen = set()
            unique_docs, unique_scores = [], []
            for doc, score in zip(docs, vector_scores):
                snippet_key = doc.page_content[:100]
                if snippet_key not in seen:
                    seen.add(snippet_key)
                    unique_docs.append(doc)
                    unique_scores.append(score)

            docs = unique_docs[:40]  # Trim to top 40 candidates
            vector_scores = unique_scores[:40]

            if not docs:
                return []

            # Re-rank documents using the cross-encoder
            rerank_scores = self.cross_encoder.predict([[query, d.page_content] for d in docs])
            fused_scores = [
                (self.alpha * vec + (1 - self.alpha) * rank, doc)
                for vec, rank, doc in zip(vector_scores, rerank_scores, docs)
            ]
            fused_scores.sort(reverse=True, key=lambda x: x[0])  # Sort by final score

            return [doc for _, doc in fused_scores[:20]]  # Return top 20 final results

    return SimpleFusionRetriever(
        base_retriever=vector_retriever,
        cross_encoder=cross_encoder,
        alpha=alpha,
        inv_index=inverted_index,
        store=index
    )
