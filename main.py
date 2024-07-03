import os
import streamlit as st
import time
import fitz  # PyMuPDF
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
import re

# Ensure that the OPENAI_API_KEY is set
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

st.title("News Research Chat")
st.sidebar.title("Upload Content")

# Initialize session state
if 'urls' not in st.session_state:
    st.session_state.urls = ['']
if 'pdf_files' not in st.session_state:
    st.session_state.pdf_files = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to add a new URL input
def add_url():
    st.session_state.urls.append('')

# Display URL input fields dynamically
for i in range(len(st.session_state.urls)):
    st.session_state.urls[i] = st.sidebar.text_input(f"URL {i+1}", value=st.session_state.urls[i], key=f"url_{i}")

# PDF file uploader
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
if uploaded_files:
    st.session_state.pdf_files = uploaded_files

if st.sidebar.button("Add another URL"):
    add_url()

process_content_clicked = st.sidebar.button("Process Content")

main_placeholder = st.empty()
chat_model = ChatOpenAI(temperature=0.7, max_tokens=500)

def extract_text_from_pdf(pdf):
    doc = fitz.open(stream=pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters except punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text

if process_content_clicked:
    documents = []
    
    # Process URLs
    valid_urls = [url for url in st.session_state.urls if url.strip()]
    if valid_urls:
        main_placeholder.text("Processing URLs...")
        loader = UnstructuredURLLoader(urls=valid_urls)
        url_documents = loader.load()
        for doc in url_documents:
            doc.page_content = preprocess_text(doc.page_content)
        documents.extend(url_documents)
        main_placeholder.text(f"Processed {len(url_documents)} URLs")
        time.sleep(1)

    # Process PDFs
    if st.session_state.pdf_files:
        main_placeholder.text("Processing PDFs...")
        for pdf_file in st.session_state.pdf_files:
            pdf_text = extract_text_from_pdf(pdf_file)
            pdf_text = preprocess_text(pdf_text)
            pdf_document = Document(page_content=pdf_text, metadata={"source": pdf_file.name})
            documents.append(pdf_document)
        main_placeholder.text(f"Processed {len(st.session_state.pdf_files)} PDFs")
        time.sleep(1)

    if documents:
        main_placeholder.text("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        docs = text_splitter.split_documents(documents)

        main_placeholder.text("Building vector store...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.session_state.vectorstore = vectorstore

        # Initialize conversation memory
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        main_placeholder.text(f"Processing complete. Total chunks: {len(docs)}")
        time.sleep(2)
    else:
        st.warning("No content to process. Please add URLs or upload PDF files.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if st.session_state.vectorstore:
            chain = ConversationalRetrievalChain.from_llm(
                llm=chat_model,
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=st.session_state.memory
            )
            result = chain({"question": prompt})
            full_response = result['answer']

            # Simulate stream of response with milliseconds delay
            for chunk in full_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        else:
            full_response = "Please process some content before starting the chat."
            message_placeholder.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.memory.clear()