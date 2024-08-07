import os
import streamlit as st
import time
import fitz 
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import re

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="📰 ResearchMate", page_icon="📰", layout="wide")

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; color: inherit !important; animation: pulse 2s infinite;'>
        📰 ResearchMate
    </h1>
    <style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📄 Upload Content")

if 'urls' not in st.session_state:
    st.session_state.urls = ['']
if 'pdf_files' not in st.session_state:
    st.session_state.pdf_files = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def add_url():
    st.session_state.urls.append('')

with st.sidebar.expander("Add URLs", expanded=True):
    for i in range(len(st.session_state.urls)):
        st.session_state.urls[i] = st.text_input(f"URL {i+1}", value=st.session_state.urls[i], key=f"url_{i}")

if st.sidebar.button("➕ Add another URL"):
    add_url()

uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
if uploaded_files:
    st.session_state.pdf_files = uploaded_files

process_content_clicked = st.sidebar.button("🔍 Process Content")

main_placeholder = st.empty()
chat_model = ChatOpenAI(temperature=0.7, max_tokens=500)

def extract_text_from_pdf(pdf):
    doc = fitz.open(stream=pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text

if process_content_clicked:
    documents = []
    
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

template = """You are an AI assistant tasked with answering questions based on the given context. 
Use the information provided in the context to answer the question concisely and avoid repetition. 
I am saying this strictly that If the answer cannot be found in the context, simply state that you don't have enough information to answer accurately.

Context: {context}
Question: {question}
Answer: """

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

if 'qa' not in st.session_state and st.session_state.vectorstore is not None:
    st.session_state.qa = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

for message in st.session_state.chat_history:
    message_class = "user-message" if message["role"] == "user" else "bot-message"
    with st.container():
        st.markdown(f"""
            <div class='chat-message {message_class}'>
                <p>{message['content']}</p>
            </div>
        """, unsafe_allow_html=True)

if prompt := st.chat_input("💬 What would you like to know?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.container():
        st.markdown(f"""
            <div class='chat-message user-message'>
                <p>{prompt}</p>
            </div>
        """, unsafe_allow_html=True)

    with st.container():
        message_placeholder = st.empty()
        full_response = ""

        if 'qa' in st.session_state:
            result = st.session_state.qa({"query": prompt})
            answer = result['result']
            source_documents = result['source_documents']

            sources = set([doc.metadata['source'] for doc in source_documents])
            sources_text = "\n\nSources: " + ", ".join(sources)

            full_response = answer + sources_text

            for i in range(len(full_response)):
                time.sleep(0.01)
                message_placeholder.markdown(f"""
                    <div class='chat-message bot-message'>
                        <p>{full_response[:i]}▌</p>
                    </div>
                """, unsafe_allow_html=True)
            
            message_placeholder.markdown(f"""
                <div class='chat-message bot-message'>
                    <p>{full_response}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            full_response = "Please process some content before starting the chat."
            message_placeholder.markdown(f"""
                <div class='chat-message bot-message'>
                    <p>{full_response}</p>
                </div>
            """, unsafe_allow_html=True)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

st.sidebar.write("## Manage Session")
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")

if st.sidebar.button("🗑️ Clear Processed Content"):
    st.session_state.vectorstore = None
    if 'qa' in st.session_state:
        del st.session_state.qa
    st.success("Processed content cleared. You can now upload new content.")
