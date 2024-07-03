import os
import streamlit as st
import time
import fitz  # PyMuPDF
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import re

# Ensure that the OPENAI_API_KEY is set
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

st.title("News Research Tool")
st.sidebar.title("Upload Content")

# Initialize session state
if 'urls' not in st.session_state:
    st.session_state.urls = ['']
if 'pdf_files' not in st.session_state:
    st.session_state.pdf_files = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

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
llm = OpenAI(temperature=0.7, max_tokens=500)

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

        main_placeholder.text(f"Processing complete. Total chunks: {len(docs)}")
        time.sleep(2)
    else:
        st.warning("No content to process. Please add URLs or upload PDF files.")

query = main_placeholder.text_input("Question:")

if query:
    if st.session_state.vectorstore:
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, 
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.subheader(result["answer"])
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
    else:
        st.warning("Please process some content before asking questions.")