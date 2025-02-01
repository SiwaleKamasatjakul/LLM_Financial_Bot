import streamlit as st
from tools.url_processor import URLProcessor
from tools.query_handler import QueryHandler


class StreamlitInterface:
    def __init__(self):
        self.url_processor = URLProcessor()
        self.query_handler = QueryHandler()

    def run(self):
        """Run the Streamlit interface."""
        st.title("News Research Tool")
        st.sidebar.title("News Article URLs")

        # Input URLs
        urls = self.get_urls_from_sidebar()

        # Button to process URLs
        if st.sidebar.button("Process URLs"):
            self.process_urls(urls)

        # Query input
        query = st.text_input("Enter your question:")
        if query:
            self.process_query(query)

    def get_urls_from_sidebar(self):
        """Get URLs from Streamlit sidebar."""
        urls = []
        for i in range(3):
            url = st.sidebar.text_input(f"URL {i + 1}")
            urls.append(url)
        return urls

    def process_urls(self, urls):
        """Process URLs and create FAISS index."""
        try:
            st.text("Processing URLs...")
            self.url_processor.process_urls(urls)
            st.success("URLs processed and FAISS index created successfully.")
        except Exception as e:
            st.error(f"Error processing URLs: {e}")

    def process_query(self, query):
        """Process the query and retrieve an answer."""
        try:
            self.query_handler.load_vectorstore()  # Load vector store
            result = self.query_handler.query(query)  # Get result

            st.header("Answer")
            st.subheader(result["answer"])

            # Display source URLs
            if result["sources"]:
                st.header("Sources")
                for source in result["sources"]:
                    st.write(source)

        except Exception as e:
            st.error(f"Error retrieving answer: {e}")
