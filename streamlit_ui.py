import streamlit as st
import requests
import time

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Invoice QA Chat", layout="wide")
st.title("Invoice QA Assistant")

st.sidebar.header("Upload Invoices")
uploaded_files = st.sidebar.file_uploader("Choose PDF/CSV/TSV files", type=["pdf", "csv", "tsv"], accept_multiple_files=True)

if st.sidebar.button("Upload & Process") and uploaded_files:
    with st.spinner("Uploading and processing files..."):
        files = [("files", (f.name, f.read(), f"application/{f.name.split('.')[-1]}")) for f in uploaded_files]
        res = requests.post(f"{BACKEND_URL}/load-documents", files=files)
        if res.ok:
            st.sidebar.success("Uploaded. Indexing in background...")
        else:
            st.sidebar.error("Upload failed.")

# Status checker
status_btn = st.sidebar.button("Check Status")
if status_btn:
    res = requests.get(f"{BACKEND_URL}/status")
    data = res.json()
    st.sidebar.info(f"Status: {data['status']}")
    if data['status'] == 'ready':
        st.sidebar.success(data['message'])
    else:
        st.sidebar.warning(data['message'])

# Chat Interface
st.subheader("Ask Questions About Your Invoices")
query = st.text_input("Type your question:", placeholder="e.g., How many invoices are billed to London?")

if st.button("Ask") and query:
    with st.spinner("Getting answer from the assistant..."):
        res = requests.post(f"{BACKEND_URL}/query", json={"question": query})
        if res.ok:
            result = res.json()
            st.success(result['answer'])
        else:
            st.error("Error in answering the question. Check if files are uploaded and status is 'ready'.")

# Reset chat history
if st.button("🧹 Reset Chat History"):
    requests.post(f"{BACKEND_URL}/reset-history")
    st.success("Chat history cleared.")
