import os
import pickle
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import HuggingFaceHub

# Load environment variables
load_dotenv()


class QueryHandler:
    def __init__(self, file_path="faiss_store_openai.pkl"):
        self.file_path = file_path
        self.vectorstore = None
        self.llm = None

    def load_vectorstore(self):
        """Load the FAISS vector store from file."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as f:
                self.vectorstore = pickle.load(f)
        else:
            raise FileNotFoundError("FAISS index file not found. Please process URLs first.")

    def initialize_llm(self):
        """Initialize the LLM using Hugging Face API."""
        huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not huggingface_api_token:
            raise ValueError("Hugging Face API token is missing. Please check your .env file.")
        self.llm = HuggingFaceHub(repo_id="google/flan-t5-base", huggingfacehub_api_token=huggingface_api_token)

    def query(self, question):
        """Query the vector store and return the result, including source URLs."""
        if not self.vectorstore:
            raise ValueError("Vector store is not loaded. Please load or create it first.")
        if not self.llm:
            self.initialize_llm()

        # Create chain and query
        chain = RetrievalQAWithSourcesChain.from_llm(llm=self.llm, retriever=self.vectorstore.as_retriever())
        result = chain({"question": question}, return_only_outputs=True)

        # Extract the answer and sources
        answer = result.get("answer", "No answer found.")
        sources = result.get("sources", [])

        return {"answer": answer, "sources": sources}