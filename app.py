import streamlit as st
import subprocess
import os
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import time

# Load environment variables
model = os.environ.get("MODEL", "mistral")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

# Function to initialize the model and components
def initialize_model():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = Ollama(model=model, callbacks=callbacks)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

# Function to get response from the model
def get_response(qa, query):
    res = qa(query)
    answer = res['result']
    source_documents = res['source_documents']
    return answer, source_documents

# Function to execute ingest.py script
def execute_ingest_script():
    process = subprocess.Popen(["python", "ingest.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if stderr:
        st.error(stderr.decode("utf-8"))
    else:
        st.success(stdout.decode("utf-8"))

# Main function to create the chatbot interface
def main():
    st.title("Chat with Docs")
    st.write("Welcome to the Chat with Docs interface. Ask any question and get answers from your documents.")

    # Display accepted file extensions
    with st.expander("Show Accepted File Extensions"):
        st.write("""
        - .csv
        - .doc, .docx
        - .enex
        - .eml
        - .epub
        - .html
        - .md
        - .odt
        - .pdf
        - .ppt, .pptx
        - .txt
        """)

    # Initialize the model and components
    qa = initialize_model()

    # File upload feature
    uploaded_files = st.file_uploader("Upload Files", type=["csv", "doc", "docx", "enex", "eml", "epub", "html", "md", "odt", "pdf", "ppt", "pptx", "txt"], accept_multiple_files=True)

    # Store uploaded files in source_documents directory
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("source_documents", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        st.success("Files uploaded successfully!")

    # Button to run ingest.py script
    if st.button("Run Ingest Script"):
        execute_ingest_script()

    # Chatbot interface
    query = st.text_input("You: ", "", key="input_field")

    # Execute "Send" button click event when Enter key is pressed
    if st.session_state.enter_pressed:
        st.session_state.enter_pressed = False
        st.button("Send")

    if st.button("Send") or st.session_state.enter_pressed:
        if query:
            st.write("Bot: Processing...")
            answer, source_documents = get_response(qa, query)
            st.write(f"Bot: {answer}")

            if source_documents:
                st.write("**Source Documents:**")
                for document in source_documents:
                    st.write(f"*{document.metadata['source']}:*")
                    st.write(document.page_content)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
