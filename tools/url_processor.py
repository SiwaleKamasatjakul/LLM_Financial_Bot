import os
import pickle
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  # Correct import
from langchain.vectorstores import FAISS

class URLProcessor:
    def __init__(self, file_path="faiss_store_openai.pkl"):
        self.file_path = file_path

    def process_urls(self, urls):
        """Load data from URLs and create a FAISS vector store."""
        # Load and process data
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Add the URLs as metadata to the documents
        for i, doc in enumerate(data):
            doc.metadata['source_url'] = urls[i]

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)

        # Generate embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Save the FAISS index
        with open(self.file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        return vectorstore